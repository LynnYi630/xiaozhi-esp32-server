# xiaozhi-server/plugins_func/functions/get_stock_price.py

import random  # 假设我们用随机数模拟股票价格
from plugins_func.register import register_function, ToolType, ActionResponse, Action

# --- 核心要素 1: 工具描述 ---
# 这是给大语言模型(LLM)看的“使用说明书”。
# LLM会根据这里的 name, description 和 parameters 来决定何时以及如何调用这个工具。
GET_STOCK_PRICE_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "获取指定股票代码的最新价格信息。",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_code": {
                    "type": "string",
                    "description": "股票的代码，例如 '600519' 代表贵州茅台。",
                }
            },
            "required": ["stock_code"],
        },
    },
}

# --- 核心要素 2 & 3: 注册器 + 工具函数 ---
# @register_function 是注册器，它会将下面的函数注册到系统中。
# 第一个参数 "get_stock_price" 是工具的唯一ID，必须和上面描述中的 name 一致。
# 第二个参数是上面定义的工具描述。
# 第三个参数 ToolType.SYSTEM_CTL 是工具类型，表示这是一个系统控制类工具。
@register_function("get_stock_price", GET_STOCK_PRICE_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def get_stock_price(conn, stock_code: str):
    """
    这个函数是工具的具体实现。
    当LLM决定调用此工具时，这个函数就会被执行。
    参数 conn 可以用来获取系统配置，stock_code 则是从LLM的调用指令中解析出来的参数。
    """
    
    # 1. 在这里编写获取股票价格的真实逻辑
    # (例如: 调用 requests 访问一个股票API)
    # 为了演示，我们这里只生成一个随机价格
    try:
        # 模拟API调用
        price = round(random.uniform(50.0, 500.0), 2)
        stock_report = f"您查询的股票 {stock_code} 当前价格为 {price} 元。"
        
        # 2. 准备返回结果
        # ActionResponse 是标准的返回格式，它告诉系统下一步该做什么。
        # Action.REQLLM: 表示“把我的执行结果(result)再发给LLM，让它用更友好的方式说出来”。
        # result: 这就是你的函数执行结果，这里是包含价格的字符串。
        return ActionResponse(Action.REQLLM, stock_report, None)

    except Exception as e:
        # 3. 异常处理
        error_message = f"查询股票 {stock_code} 价格时发生错误: {e}"
        return ActionResponse(Action.ERROR, None, error_message)