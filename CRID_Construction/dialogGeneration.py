import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI
from datetime import datetime
import logging
from dialogGeneration_str_config import *
import re
from itertools import cycle

# 配置日志记录的基本参数
logging.basicConfig(
    level=logging.INFO,  # 设置记录级别为DEBUG（最低级别）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    filemode='a'  # 文件写入模式：'a'表示追加，'w'表示覆盖
)

logger = logging.getLogger(__name__)

logger.info('开始执行')       


res = {}
input_file_path = ''
output_file_path = ''
DIG_NUM = 3
MAX_ROUNDS = 40
RETRY_DELAY = 1
MAX_ATTEMPTS_PER_CLIENT = 10
MAX_WORKERS = 500

client_pool = [
    OpenAI(api_key="", base_url=""),
]
client_cycle = cycle(client_pool)

client_index_lock = threading.Lock()
client_index = 0

def filter_format(s):
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and start < end:
        return s[start:end+1]
    
    raise ValueError("字符串中找不到匹配的 '{' 和 '}'")

def filter_state(s):
    result = re.sub(r"^<状态\[.*?\]>\s*", "", s)
    return result

def get_next_client():
    global client_index
    with client_index_lock:
        client_index = (client_index + 1) % len(client_pool)
    
    return


def Deeseek_API_request(messages):
    attempts = 0
    tried_clients = 0

    while tried_clients < (3 * len(client_pool)):
        client = next(client_cycle)

        for _ in range(MAX_ATTEMPTS_PER_CLIENT):
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    response_format={'type': 'json_object'},
                    temperature=0.8,
                    stream=False
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.warning(response.choices[0].message.content)
                logger.warning(e)
                attempts += 1
                time.sleep(RETRY_DELAY * attempts * (tried_clients + 1))  # exponential-ish backoff

        # 当前 client 尝试失败，切换到下一个
        get_next_client()
        tried_clients += 1

    logger.info('All clients failed after multiple attempts.')
    return {'FAILED':'!!!!!!!!!!!!!!'}


def process_single_dig(case_str):
    tea_context = []
    pro_context = []
    dig = []
    rounds = 0
    Flag = True
    while True:
        rounds += 1
        if rounds >= MAX_ROUNDS: 
            logger.warning("PASS MAX_ROUNDS")
            # Flag = False
            break

        tea_messages = [
            {"role": "system", "content": tea_system_prompt},
            {"role": "user", "content": tea_user_prompt.format(case=case_str, tea_context=tea_context)},
        ]
        tea_response = Deeseek_API_request(tea_messages)
        if '教师' not in tea_response:
            logger.warning(tea_response)
            tea_response = {'教师': next(iter(tea_response.values()))}
            # Flag = False
            # break

        pro_context.append({'教师': tea_response['教师']})
        dig.append({'教师': tea_response['教师']})
        if '<结束>' in list(tea_response.values())[0]: break

        pro_messages = [
            {"role": "system", "content": pro_system_prompt},
            {"role": "user", "content": pro_user_prompt.format(case=case_str, pro_context=pro_context)},
        ]
        pro_response = Deeseek_API_request(pro_messages)
        if '专家' not in pro_response:
            logger.warning(pro_response)
            pro_response = {'专家': next(iter(pro_response.values()))}
            # Flag = False
            # break

        try:
            notation, pro_response_content = pro_response['专家'].split(">", 1)
        except ValueError:
            logger.warning(pro_response)
            notation = "<无违反准则>"
            pro_response_content = ""

        if (notation + '>') == '<无违反准则>':
            tea_context.append({'教师': tea_response['教师']})
        else:
            tea_context.append({'教师': pro_response_content})
        pro_context[-1]['专家'] = pro_response['专家']
        dig[-1]['专家'] = pro_response['专家']

        stu_messages = [
            {"role": "system", "content": stu_system_prompt},
            {"role": "user", "content": stu_user_prompt.format(case = case_str, stu_context=tea_context)},
        ]
        stu_response = Deeseek_API_request(stu_messages)
        if '学生' not in stu_response:
            logger.warning(stu_response)
            stu_response = {'学生': next(iter(stu_response.values()))}

        tea_context[-1]['学生'] = filter_state(stu_response['学生'])
        pro_context[-1]['学生'] = filter_state(stu_response['学生'])
        dig[-1]['学生'] = stu_response['学生']

    return dig if Flag else None



def process_dialog_case(key, val):
    logger.info(key)
    case_str = json.dumps(val["案例拆分"], ensure_ascii=False, indent=2)

    dig_result = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor: 
        futures = [executor.submit(process_single_dig, case_str) for _ in range(DIG_NUM)]
        idx = 0
        for future in as_completed(futures):
            dig = future.result()
            if dig:  # 成功生成才记录
                dig_result[f"{key}_{idx}"] = dig
                idx += 1
    return dig_result

def multithread_generate_dialogs(data_dict, max_workers=MAX_WORKERS):
    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_dialog_case, key, val) for key, val in data_dict.items()]
        for future in as_completed(futures):
            dialog_items = future.result()
            result.update(dialog_items)
    return result

# === 读取数据并执行 ===
with open(input_file_path, 'r', encoding='utf-8') as f:
    input_data = json.load(f)


logger.info('file load success!')

dialog_results = multithread_generate_dialogs(input_data, max_workers=MAX_WORKERS)

logger.info('结束执行')   

with open(output_file_path, "w") as f:
    f.write(json.dumps(dialog_results, ensure_ascii=False, indent=2))
