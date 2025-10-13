import json
import time
import threading
from openai import OpenAI
from datetime import datetime
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
import os


# 配置日志记录的基本参数
logging.basicConfig(
    level=logging.INFO,  # 设置记录级别为DEBUG（最低级别）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    filemode='a'  # 文件写入模式：'a'表示追加，'w'表示覆盖
)

logger = logging.getLogger(__name__)

logger.info('开始执行')  
    

input_file_path = ''
output_file_path = ""

RETRY_DELAY = 1
MAX_ATTEMPTS_PER_CLIENT = 10
MAX_WORKERS = 50


client_pool = [
    OpenAI(api_key="", base_url=""),
]
client_cycle = cycle(client_pool)


system_prompt = """你的任务是将格式非规范的病例文本转换为结构化的标准格式。请仔细阅读以下原始病例文本，并按照后续要求完成处理：
{case}

# 结构化信息输出
请基于原始病例文本内容，严格按照以下九个步骤完成信息提取与标准化：
1. 案例背景：提炼患者主诉、现病史、既往史中的关键信息，以及体格检查中的异常表现，形成简明背景描述。不包含任何辅助检查或诊断信息。
2. 病症关注：基于上述案例背景，分析并提取与最终诊断高度相关的症状或体征，并明确疾病所涉及的人体系统或所属的临床医学范畴，不涉及对具体疾病类型的推断或初步诊断。
3. 初步诊断：根据案例背景，提出或生成合理、可追溯、具有线索支持的初步诊断内容，仅供参考，无需列出具体诊断依据或鉴别诊断内容。
4. 辅助检查：从整个病例文本中筛选出与前三步内容密切相关的检查项目（如心电图、CT、血常规、心脏超声等），确保其具有充分的临床依据。
5. 检查结果：按顺序对应上述检查项目，列出各自的检查结果，不得直接泄露最终诊断信息。
6. 诊断确认：根据病例文本，提取最终诊断结论。诊断应有案例背景和检查结果中的客观依据作为支撑，避免无根据的推测，仅限于与本次病症表现直接相关的疾病（不包括仅属既往史的疾病）。无需列出具体诊断依据。
7. 治疗方案：一一对应诊断结果，从原始文本中中提取有针对性的治疗措施。
8. 人文关怀：基于所属科室特点，推演一个该病例可能存在的典型医患沟通问题，并给出一段具有同理心与专业表达的沟通示范语。参考如下：
    · 儿科：安心——用同理心和平静的支持来解决父母的焦虑和恐惧。例如：
        - 若家长对使用抗生素有顾虑，可沟通为：“我非常理解您对抗生素使用的顾虑……”
    · 内科：耐心——管理慢性病需要长期的理解和专注的倾听。例如：
        - 若患者因长期病情反复表达绝望时，可回应：“我们完全理解您这些年承受的痛苦和此刻的焦虑……”
    · 外科：信任——帮助患者在侵入性手术前后感到安全和自信。例如：
        - 若患者对开放性骨折清创手术感到恐惧，可解释为：“您对清创手术的担忧非常正常，面对开放性伤口任何人都会紧张……"
    · 妇产科：尊重——尊重隐私、尊严和情感敏感性，特别是在分娩和生殖健康方面。例如：
        - 如果患者表示未婚未育，担心手术会影响未来生育功能，可沟通为：“我们非常重视您对生育功能的顾虑……”
9. 分析总结：系统回顾整个诊疗过程，概括诊断依据、诊断结论、治疗策略和人文关怀要点，体现出病例分析的完整性与逻辑性。

# 请参考下方结构化示例，按同样格式输出本病例的 JSON 结构化结果：
{{
    "科室": "内科",
    "案例背景": "患者是一位65岁女性，主诉乏力、纳差、尿黄10余天，既往有服用速效伤风胶囊的药物史。",
    "病症关注": "该患者的主要症状是持续10余天的乏力、纳差，提示可能存在肝胆系统相关问题。",
    "初步诊断": "初步考虑药物性肝炎，肝硬化。",
    "辅助检查": "(1)肝功能相关血液生化检查；(2)HBV相关病毒学检查；(3)腹部影像学检查，如CT或B超；(4)血常规与凝血功能检查。",
    "检查结果": "(1)血液检查显示ALT 441U/L、AST 649U/L、TBIL 57.77μmol/L；(2)HBsAg、HBVDNA高度阳性，HBV前S1抗原阳性；(3)腹部CT提示肝右叶囊肿；(4)血小板减少，贫血，中性粒细胞升高。",
    "诊断确认": "(1)慢性乙型病毒性肝炎(2)肝硬化失代偿期(3)肝功能不全，合并脾大(4)低蛋白血症及中度贫血。",
    "治疗方案": "(1)对于乙型肝炎，给予恩替卡韦等药物进行抗病毒治疗以抑制HBV复制；(2)针对肝硬化失代偿，联合使用护肝药物、退黄治疗、营养支持和补液以改善肝功能；(3)针对肝功能不全合并脾大，定期监测脾功能、血常规，评估门脉高压相关并发症；(4)输注白蛋白纠正低蛋白血症，补充铁剂、维生素K改善贫血与凝血功能。",
    "人文关怀": "如果患者担心病情严重，情绪不稳定，我会这样安慰：我理解您此刻的担忧，面对这样的诊断，感到不安是很自然的反应。请放心，我们会一步一步陪您一起面对病情。像乙型肝炎和肝硬化这样的慢性病，确实需要长期管理，但也是可以通过规范治疗和定期复查来逐步稳定和控制的。我们会为您制定清晰的治疗计划，并在每一个阶段都认真倾听您的感受和反馈。如果您在住院或日常生活中遇到任何不适或者疑问，随时可以告诉我们，我们会耐心听您讲，尽全力帮助您。",
    "分析总结": "患者65岁女性，因乏力、纳差、尿黄就诊，检查示ALT、AST升高，HBV DNA阳性，腹部CT提示肝脏改变，确诊为乙型肝炎相关肝硬化失代偿期，给予抗病毒及支持治疗，同时加强心理疏导，安慰患者，增强治疗信心。"
}}
"""




def Deeseek_API_request(messages):
    attempts = 0
    tried_clients = 0
    
    while tried_clients < len(client_pool):
        client = next(client_cycle)

        for _ in range(MAX_ATTEMPTS_PER_CLIENT):
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    response_format={'type': 'json_object'},
                    temperature=1,
                    stream=False
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.warning(e)
                attempts += 1
                time.sleep(RETRY_DELAY * attempts * (tried_clients + 1))  # exponential-ish backoff

        tried_clients += 1

    logger.info('All clients failed after multiple attempts.')
    return 'FAILED'


def process_case(case):
    
    prompt = system_prompt.format(case = json.dumps(case, ensure_ascii=False))
    messages = [
        {"role": "system", "content": prompt},
    ]

    response = Deeseek_API_request(messages)
    return response


def multithread_process(data, max_workers=20):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_case, case["original_text"]): key
            for key, case in data.items()
        }

        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            data[key]["restructed_text"] = result
    return data


def load_data(folder):
    # 设置文件夹路径

    with open(folder, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}

    for item in data:
        result[item["id"]] = {}
        result[item["id"]]["original_text"] = item
        result[item["id"]]["original_text"].pop("id")
    
    return result


data = load_data(input_file_path)

res = multithread_process(data, max_workers=MAX_WORKERS)

with open(output_file_path, "w") as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=2))