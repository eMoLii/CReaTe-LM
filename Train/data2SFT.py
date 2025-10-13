import json
import random
import numpy as np
import copy
import re
import copy
from prompt_str import tea_system_prompt, tea_user_prompt


def filter(s):
    p = s.rfind("> ")
    return s[p + 2:].lstrip()

def filter1(s):
    p = s.find("> ")
    return s[p + 2:].lstrip()


with open('', 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(12345)
np.random.seed(12345)

out = []


for key, val in data.items():
    for k, v in val['对话'].items():         
        sample = {}
        case = copy.deepcopy(val['案例拆分'])
        patien_information = case.pop("案例背景")
        examination_results = case.pop("检查结果")
        reference_answer = json.dumps(case, ensure_ascii=False, indent=2)
        sample['system'] = tea_system_prompt.format(patien_information = patien_information, examination_results = examination_results, reference_answer = reference_answer)
        sample['conversations'] = []
        sample['conversations'].append({"from": "user", "value": tea_user_prompt})

        for i, dia in enumerate(v):
            if('专家' in dia and dia['专家'] != "<无违反准则>"):
                sample['conversations'].append({"from": "assistant", "value": filter(dia['专家'])})
            else:
                sample['conversations'].append({"from": "assistant", "value": filter1(dia['教师'])})

            if('学生' in dia):
                sample['conversations'].append({"from": "user", "value": "学生：" + filter1(dia['学生'])})

        out.append(sample)


with open('data/SFT_multirounds_dialog.json', 'w', encoding='utf-8') as f_out:
    json.dump(out, f_out, ensure_ascii=False, indent=2)
