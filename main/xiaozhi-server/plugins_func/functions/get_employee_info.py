import sys
import os
from pypinyin import pinyin, Style
# 引入高性能编辑距离库
import Levenshtein  
from sqlalchemy import create_engine, Column, Integer, String, or_
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.config_loader import load_config

# 加载配置
config = load_config()
MYSQL_CONFIG = config.get("mysql", {})

TAG = __name__
logger = setup_logging()

# ==============================================================================
# 1. SQLAlchemy 初始化 & 模型定义
# ==============================================================================

# 构建数据库连接 URL
DB_USER = MYSQL_CONFIG.get("user", "root")
DB_PASSWORD = MYSQL_CONFIG.get("password", "")
DB_HOST = MYSQL_CONFIG.get("host", "127.0.0.1")
DB_PORT = MYSQL_CONFIG.get("port", 3306)
DB_NAME = MYSQL_CONFIG.get("db", "")

# 确保已安装 pymysql: pip install pymysql
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# 创建引擎和会话工厂
engine = create_engine(
    DATABASE_URL,
    pool_recycle=3600,
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义 Employee 模型
class Employee(Base):
    __tablename__ = 'person_info'

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_name = Column(String(50), nullable=False)
    phone_number = Column(String(20))
    office_address = Column(String(255))
    job_title = Column(String(100))
    department = Column(String(100))
    supervisor = Column(String(50))
    name_pinyin = Column(String(100))

    def to_dict(self):
        """转字典，方便后续处理"""
        return {
            "name": self.person_name,
            "job_title": self.job_title,
            "department": self.department,
            "phone": self.phone_number,
            "office_address": self.office_address,
            "name_pinyin": self.name_pinyin,
            "supervisor": self.supervisor
        }

# ==============================================================================
# 2. 工具描述 (Schema)
# ==============================================================================
GET_EMPLOYEE_DESC = {
    "type": "function",
    "function": {
        "name": "get_employee_info",

        "description": (
            "【高优先级】用于查询公司员工的联系方式、办公位置等信息。"

            "当用户询问“xx在哪”、“xx的电话”、“xx的办公室在哪”、“谁是xx部门的经理”、“我要找xx”、“我要找一下xx”时，必须立即调用此工具。"

            "注意：【关键约束】在提取参数时，你只能使用 name_keyword, department, job_title, office_address_keyword 这四个定义的参数。严禁将用户问题中描述查询目标的词语（如“电话”、“上级”、“办公室”、“是谁”）提取为额外的参数（例如 'query' 或其他未定义的键）。"

            "本工具支持通过【姓名】、【职务】、【部门】或【办公地址】等**任意组合**进行查询。"

            "注意：如果用户提供了姓名，请务必优先使用姓名进行查询。如果姓名未命中，再考虑职务、部门等其他条件进行辅助筛选。"

            "**【重要示例】** 如果用户问：'谁是黄耀科的上级？'，你**必须且只能**提取 name_keyword='黄耀科'，其他参数（包括 'query', '上级' 等）一律留空。"

            "如果用户只提供了职称（如'韩总'、'王经理'），请不要猜测，直接请求用户提供具体的全名。"
            
            "如果用户询问职务反查姓名（如：谁是技术部总监），则提取 department='技术部' 和 job_title='总监'，name_keyword留空。"
            
            "如果用户通过地址查询姓名/电话（如：二楼213是谁），则提取 office_address_keyword='二楼213'，其余参数留空。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name_keyword": {
                    "type": "string",
                    "description": "员工姓名或姓氏（选填）。请提取用户的请求中对应的汉字。",
                },
                "department": {
                    "type": "string",
                    "description": "部门关键词（选填），用于辅助筛选。",
                },
                "job_title": {
                    "type": "string",
                    "description": "职务/职称关键词（选填），用于辅助筛选或反查。",
                },
                "office_address_keyword": { 
                    "type": "string",
                    "description": "办公地址或工位关键词（选填），如 '二楼213'。",
                }
            },
            "required": [],
        },
    },
}

# ==============================================================================
# 3. 辅助逻辑 (仅保留拼音转换)
# ==============================================================================

def get_pinyin_str(text):
    """获取文本的拼音字符串（不带声调），如 '黄' -> 'huang'"""
    if not text:
        return ""
    pins = [p[0] for p in pinyin(text, style=Style.NORMAL)]
    return "".join(pins)

# ==============================================================================
# 4. 核心工具逻辑
# ==============================================================================

@register_function("get_employee_info", GET_EMPLOYEE_DESC, ToolType.WAIT)
def get_employee_info(name_keyword: str = None, department: str = None, job_title: str = None, office_address_keyword: str = None):

    # 常用职务列表 (您可以根据公司实际职务进行扩展)
    COMMON_TITLES = ['总经理', '总监', '副总监', '工程师', '技术员']

    # 检查 LLM 是否将职务错放到了 name_keyword 中
    # 条件：1. name_keyword 存在； 2. 其他查询字段都为空； 3. name_keyword 是一个职务
    if (name_keyword and 
        name_keyword in COMMON_TITLES and
        not department and 
        not job_title and 
        not office_address_keyword):
        
        # 修正参数：将 name_keyword 视为 job_title 进行反查
        job_title = name_keyword
        name_keyword = None

    logger.bind(tag=TAG).info(f"查询员工: name='{name_keyword}', dept='{department}', title='{job_title}', address='{office_address_keyword}'")
    MAX_RESULTS = 3
    candidates = [] 
    search_method = "exact" 

    session = SessionLocal()
    try:
        # -------------------------------------------------------
        # 阶段一：精确汉字匹配 (优先)
        # -------------------------------------------------------
        if not (name_keyword or department or job_title or office_address_keyword):
            return ActionResponse(Action.RESPONSE, None, "请提供姓名、部门、职务或办公地址等至少一个查询关键词。")
        
        query = session.query(Employee)
        
        if name_keyword:
            query = session.query(Employee).filter(Employee.person_name.like(f"%{name_keyword}%"))
        
        if department:
            query = query.filter(Employee.department.like(f"%{department}%"))

        # 职务匹配 (模糊，只要包含即可)
        if job_title:
            query = query.filter(Employee.job_title == job_title)
            
        # 地址匹配 (模糊，只要包含即可) 
        if office_address_keyword:
            query = query.filter(Employee.office_address.like(f"%{office_address_keyword}%"))
        
        exact_results = query.all()
        candidates = [emp.to_dict() for emp in exact_results]
        
        # -------------------------------------------------------
        # 阶段二：拼音 Levenshtein 模糊匹配
        # -------------------------------------------------------
        if not candidates:
            logger.bind(tag=TAG).info(f"汉字未命中 '{name_keyword}'，尝试拼音编辑距离检索...")
            search_method = "pinyin_fuzzy"
            
            target_pinyin = get_pinyin_str(name_keyword)
            logger.bind(tag=TAG).info(f"拼音检索: target='{target_pinyin}'")
            
            if target_pinyin:
                # 1. 数据库宽泛召回：利用首字拼音快速过滤
                first_letter = target_pinyin[0]
                logger.bind(tag=TAG).info(f"拼音首字母: first_letter='{first_letter}'")
                pinyin_query = session.query(Employee).filter(
                    Employee.name_pinyin.like(f"{first_letter}%")
                )
                
                if department:
                    pinyin_query = pinyin_query.filter(Employee.department.like(f"%{department}%"))
                
                raw_candidates = pinyin_query.all()
                logger.bind(tag=TAG).info(f"数据库首字母 '{first_letter}' 召回记录数: {len(raw_candidates)}")
                # 2. 使用 Levenshtein 库计算编辑距离
                for emp_obj in raw_candidates:
                    emp_dict = emp_obj.to_dict()
                    
                    # 获取数据库里的拼音，并去除可能存在的空格 (容错处理)
                    db_pinyin = str(emp_dict.get('name_pinyin') or '').replace(" ", "").lower()
                    
                    # 如果数据库这行没存拼音，实时转一下救急
                    if not db_pinyin:
                        db_pinyin = get_pinyin_str(emp_dict['name'])
                    
                    # 【核心算法】：计算 用户查询的姓名拼音 和 数据库值的编辑距离
                    # distance = 0 表示完全一致
                    # distance = 1 表示差一个字母 (完美解决前后鼻音 in/ing, en/eng, z/zh 等)
                    dist = Levenshtein.distance(target_pinyin, db_pinyin)
                    
                    if dist <= 1:
                        emp_dict['_dist'] = dist  # 记录距离用于排序
                        candidates.append(emp_dict)
                        logger.bind(tag=TAG).info(f"命中候选: {emp_dict['name']} (拼音:{db_pinyin}, 距离:{dist})")

    except SQLAlchemyError as e:
        logger.bind(tag=TAG).error(f"数据库查询异常: {e}")
        return ActionResponse(Action.RESPONSE, None, "查询员工数据库时发生系统错误。")
    finally:
        session.close()

    # -------------------------------------------------------
    # 结果处理
    # -------------------------------------------------------
    if not candidates:
        msg = (
            f"抱歉，在系统中未找到姓名包含 '{name_keyword}' 的员工。"
            "如果您使用的是尊称（如'韩总'），请您告知我该员工的【全名】，以便我进行精确查询。"
        )
        if department:
            msg += f" (已限定部门: {department})"
        return ActionResponse(Action.REQLLM, msg, None)

    # -------------------------------------------------------
    # 阶段三：排序 (优先完全匹配 dist=0)
    # -------------------------------------------------------
    if search_method == "pinyin_fuzzy":
        candidates.sort(key=lambda x: x.get('_dist', 0))


    is_aggregation_query = (
        len(candidates) > 1 and 
        (department or job_title) and 
        not (name_keyword or office_address_keyword)
    )

    if is_aggregation_query:
        # 提取所有员工的姓名
        names = [emp['name'] for emp in candidates]
        
        # 构造部门/职务描述（用于回复的开头）
        subject = department if department else job_title
        
        # 构造给 LLM 的最终指令（同时满足一句话和列出姓名的要求）
        llm_instruction = (
            f"{subject}目前有{len(names)}位员工，分别是{', '.join(names)}。"
        )
        
        # 【关键】使用 Action.RESPONSE 返回最终回复，跳过后续的 JSON 序列化
        # data 字段可以填充一个摘要，方便调试和日志记录
        data_summary = f"已找到{subject} {len(names)}人：{', '.join(names)}"
        
        return ActionResponse(Action.RESPONSE, data_summary, llm_instruction)

    # -------------------------------------------------------
    # 阶段四：结果输出 (结构化 JSON 格式)
    # -------------------------------------------------------
    candidates = candidates[:MAX_RESULTS]

    # 确保在文件顶部导入了 json 库 (import json)
    import json 

    # 准备返回给 LLM 的结构化数据
    result_data = {
        "search_method": "pinyin_fuzzy" if search_method == "pinyin_fuzzy" else "exact",
        "search_keyword": name_keyword,
        "candidate_list": [
            {
                # 注意：移除 _dist 避免泄露内部算法细节
                "name": emp['name'],
                "job_title": emp.get('job_title') or '未登记',
                "department": emp.get('department') or '未登记',
                "phone": emp.get('phone') or '未登记',
                "office_address": emp.get('office_address') or '未登记',
                "supervisor": emp.get('supervisor') or '未登记'
            } 
            for emp in candidates
        ]
    }

    # 将结构化数据转换为 JSON 字符串，确保中文不乱码
    info_json = json.dumps(result_data, ensure_ascii=False, indent=2)

    # 移除复杂的回复逻辑指令，只在 message 中提醒 LLM 遵守其角色设定。
    llm_instruction = "已查询到员工信息。请根据您的‘保安机器人’角色设定（先确认身份，再提供敏感信息），组织简洁的回复。"

    return ActionResponse(Action.REQLLM, info_json, llm_instruction)