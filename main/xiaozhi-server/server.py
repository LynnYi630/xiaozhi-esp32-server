import asyncio
import websockets
import yaml
from collections import deque
from config.logger import setup_logging
# 导入你提供的模块
from core.providers.vad.silero import VADProvider
from core.providers.asr.sensevoice import ASRProvider
# from core.providers.asr.doubao import ASRProvider

# 设置日志
logger = setup_logging()

# --- 新增：定义滑动窗口的大小 ---
# 存储VAD触发前多少个音频包。每个包约60ms，10个包就是约0.6秒。
# 这个值可以根据需要调整，如果说话前缀还是丢失就调大一点。
PRE_BUFFER_PACKET_COUNT = 10

# 加载配置
with open("act_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# --- 全局初始化模型 ---
# 这样可以避免每次连接都重新加载模型，节省时间和内存
try:
    logger.bind(tag=__name__).info("正在加载VAD模型...")
    VAD_PROVIDER = VADProvider(config['vad'])
    logger.bind(tag=__name__).info("VAD模型加载成功.")

    logger.bind(tag=__name__).info("正在加载ASR模型...")
    # 注意：sense_voice.py的构造函数需要两个参数
    ASR_PROVIDER = ASRProvider(config=config['asr'], delete_audio_file=config['asr']['delete_audio_file'])
    logger.bind(tag=__name__).info("ASR模型加载成功.")
except Exception as e:
    logger.bind(tag=__name__).error(f"模型加载失败: {e}", exc_info=True)
    exit(1)


class ConnectionState:
    """用于存储每个WebSocket连接的状态"""
    def __init__(self, websocket):
        self.ws = websocket
        # VAD状态属性 (参照silero.py中的conn对象)
        self.client_audio_buffer = bytearray()
        self.client_voice_window = deque(maxlen=15) # 使用deque作为滑动窗口
        self.last_is_voice = False
        self.client_have_voice = False
        self.client_voice_stop = False
        self.speech_start_time = 0.0
        self.last_activity_time = 0.0
        
        # ASR音频缓存
        # self.asr_audio_buffer = []
        # --- 修改：将asr_audio_buffer改为一个固定长度的deque，作为滑动窗口 ---
        self.asr_audio_buffer = deque(maxlen=PRE_BUFFER_PACKET_COUNT)
        self.session_id = str(websocket.id)
        
        # 绑定一个简化的logger
        class ConnLogger:
            def bind(self, tag):
                return self
            def info(self, msg):
                logger.info(f"[{self.session_id}] {msg}")
        self.logger = ConnLogger()
        self.logger.session_id = self.session_id


async def process_asr(state: ConnectionState):
    """当语音停止时，异步处理ASR"""
    MIN_AUDIO_PACKETS = 10 
    if len(state.asr_audio_buffer) < MIN_AUDIO_PACKETS:
        logger.info(f"[{state.session_id}] 音频片段过短 ({len(state.asr_audio_buffer)}包)，已忽略。")
        # --- 修改：即使忽略也要清空，为下一次滑动窗口做准备 ---
        state.asr_audio_buffer = deque(maxlen=PRE_BUFFER_PACKET_COUNT)
        return

    logger.info(f"[{state.session_id}] 检测到语音结束，开始进行ASR识别。音频包数量: {len(state.asr_audio_buffer)}")
    
    # 复制缓冲区内容用于处理，然后清空
    # audio_task = state.asr_audio_buffer.copy()
    # state.asr_audio_buffer.clear()
    # --- 修改：将deque转换为list进行处理 ---
    audio_task = list(state.asr_audio_buffer)
    state.asr_audio_buffer = deque(maxlen=PRE_BUFFER_PACKET_COUNT) # 重置为空的滑动窗口

    try:
        # 调用SenseVoice进行识别
        # speech_to_text 是一个 async function
        text, _ = await ASR_PROVIDER.speech_to_text(audio_task, state.session_id, "opus")
        
        if text and text.strip():
            logger.info(f"[{state.session_id}] ASR识别结果: {text}")
            await state.ws.send(text)
        else:
            logger.info(f"[{state.session_id}] ASR识别结果为空。")

    except Exception as e:
        logger.error(f"[{state.session_id}] ASR处理失败: {e}", exc_info=True)


async def handler(websocket):
    """处理每个WebSocket连接"""
    state = ConnectionState(websocket)
    logger.info(f"客户端已连接: {state.session_id}")

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue
            
            # # 使用VAD进行语音活动检测
            # # is_vad函数会修改state对象内部的状态
            # is_currently_speaking = VAD_PROVIDER.is_vad(state, message)

            # if is_currently_speaking:
            #     state.asr_audio_buffer.append(message)

            # --- 这是核心逻辑修改 ---

            # 记录VAD触发前的状态
            was_speaking = state.client_have_voice

            # 运行VAD检测
            is_currently_speaking = VAD_PROVIDER.is_vad(state, message)
            
            # 当VAD首次从“无语音”变为“有语音”的瞬间
            if is_currently_speaking and not was_speaking:
                logger.info(f"[{state.session_id}] VAD 触发，预缓存队列将转为录音队列。")
                # 将滑动窗口(deque)转换为一个无长度限制的普通列表，开始正式录制
                state.asr_audio_buffer = list(state.asr_audio_buffer)

            # 无论VAD是否触发，都将最新的音频包加入队列
            # - 如果未触发，它是一个固定长度的滑动窗口(deque)
            # - 如果已触发，它是一个无限长度的列表(list)
            state.asr_audio_buffer.append(message)
            
            # --- 核心逻辑修改结束 ---


            # 检查VAD模块是否标记了语音停止
            if state.client_voice_stop:
                # 触发ASR处理
                asyncio.create_task(process_asr(state))
                # 重置VAD状态以准备下一次检测
                state.client_voice_stop = False
                state.client_have_voice = False
                state.speech_start_time = 0.0
                state.last_is_voice = False
                state.client_voice_window.clear()


    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"客户端 {state.session_id} 断开连接: {e.reason}")
    except Exception as e:
        logger.error(f"[{state.session_id}] 连接处理异常: {e}", exc_info=True)
    finally:
        logger.info(f"客户端 {state.session_id} 会话结束。")


async def main():
    host = config['server']['host']
    port = config['server']['port']
    logger.bind(tag=__name__).info(f"WebSocket服务器正在启动，监听地址 ws://{host}:{port}")
    
    # 注意：需要增加max_size来支持更大的音频流负载
    async with websockets.serve(handler, host, port, max_size=10*1024*1024):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务器已手动关闭。")