import json


def convert_json_array_to_json(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    json_list = []
    output_type = "Output (a)"  # 初始设置为 Output (a)

    for item in data:
        if output_type == "Output (a)":
            output = "Output (a)"
            item["response_1"] = item["chosen"]
            item["response_2"] = item["rejected"]
            output_type = "Output (b)"  # 下次分配 Output (b)
        else:
            output = "Output (b)"
            item["response_1"] = item["rejected"]
            item["response_2"] = item["chosen"]
            output_type = "Output (a)"  # 下次分配 Output (a)

        json_object = {
            "instruction": f"""Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.

Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.

# Instruction:
{item["instruction"]}

# Output (a):
{item["response_1"]}

# Output (b):
{item["response_2"]}

# Which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)":""",
            "input": "",
            "system": "You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.",
            "output": output
        }
        json_list.append(json_object)

    with open(output_file, 'w') as file:
        json.dump(json_list, file, indent=4)


input_file_path = '../../data/hh-rlhf_dpo.json'
output_file_path = '../../data/hh-rlhf_et.json'

convert_json_array_to_json(input_file_path, output_file_path)
print("Conversion complete.")
