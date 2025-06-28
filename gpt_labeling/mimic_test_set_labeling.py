import fnmatch
import os
import openai  # for calling the OpenAI API
from numpy import dot  # for cosine similarity
import numpy as np
import json
import backoff
import time
from collections import deque

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key ="YOUR API"
path = "MIMIC TEST PATH HERE"

def folder_load(backup = 0, saved_deque = None):
    if backup == 0:
        folder_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        folder_deque = deque(folder_list)
        print(len(folder_list))
    else:
        saved_deque = np.load(saved_deque)
        folder_list = list(saved_deque)
        folder_deque = deque(folder_list)

    return folder_list, folder_deque

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def json_gpt(input: str):
    completion = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You will be provided with a emr sentence."},
            {"role": "user", "content": input},
        ],
        temperature=0.8,
    )
    text = completion.choices[0].message.content
    parsed = json.loads(text)
    return parsed

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


if __name__ == "__main__":
    start = time.time()

    folder_count = 0
    dataset = []
    dataset1 = []
    dataset2 = []

    try:
        trouble_file = list(np.load('trouble_para_test.npy'))
    except:
        trouble_file = []

    folder_list, folder_deque = folder_load(backup = 1, saved_deque = 'remain_folders_test.npy')
    print(len(folder_list))

    for folder_name in folder_list:
        if folder_count >= 700:
           break

        folder_count += 1
        print(folder_count,"from test")
        for file in os.listdir(os.path.join(path, folder_name)):
            if fnmatch.fnmatch(file, '*.txt'):

                temp = []
                temp1 = []
                temp2 = []

                ## Data Load
                folder_path = os.path.join(path, folder_name)
                txt_file = open(os.path.join(folder_path, file), "r")
                Emr_data = txt_file.read()
                # print(Emr_data)

                ## Split the Emr dataset
                Emr_split = Emr_data.split(".")
                Emr_split_data = Emr_split[:len(Emr_split) - 1]

                for i in range(len(Emr_split_data)):
                    query = f"""Use the below sentence to answer the subsequent question.
                    Emr_report:
                    \"\"\"
                    {Emr_split_data[i]}
                    \"\"\"
                    Question: Does the patient have the specific disease in the chest based on the provied EMR report's sentence? 
                    Answer form should be JSON object like following script. The JSON object has two key, "Result", and "Explanation".
                    For [Result], if the sentence doesn't have enough information or evidence to classify, you should return "Uncertain". 
                    If the sentence has the clear evidence that indicates absence of any abnormalities in chest, you should answer "No". 
                    If the sentence has the clear observational evidence that indicates presence of any abnormalities in chest (only for present), you should answer "Yes". 

                    For [Explanation], you should give a sentence more than 40 letters and less than 60 letters which explain the reason about why you choose those answers. You should elucidating the rationale behind your choice, not a direct repetition, of the input text.
                    [Result] : Uncertain / No / Yes
                    """
                    success_flag = 0
                    error_count = 0

                    while success_flag == 0 and error_count <= 50:
                        try:
                            response = json_gpt(query)
                            response["Context"] = Emr_split_data[i]
                            context_emb = get_embedding(response["Context"])
                            explanation_emb = get_embedding(response["Explanation"])
                            cosine_sim = (dot(context_emb, explanation_emb) + 1) / 2
                            response["confidence"] = cosine_sim
                            response["text_file"] = file
                            response["sentence_index"] = i
                            temp.append(response)
                            success_flag = 1
                            #print(response)


                        except:
                            time.sleep(10)
                            print("error occured in chatgpt server")
                            error_count += 1
                            continue

                    ## cannot collect the data on second loop
                    if error_count > 50:
                        trouble_file.append(file)
                        break

                    success_flag1 = 0
                    while success_flag1 == 0 and error_count <= 50:
                        try:
                            response = json_gpt(query)
                            response["Context"] = Emr_split_data[i]
                            context_emb = get_embedding(response["Context"])
                            explanation_emb = get_embedding(response["Explanation"])
                            cosine_sim = (dot(context_emb, explanation_emb) + 1) / 2
                            response["confidence"] = cosine_sim
                            response["text_file"] = file
                            response["sentence_index"] = i
                            temp1.append(response)
                            success_flag1 = 1
                            #print(response)

                        except:
                            time.sleep(10)
                            print("error occured in chatgpt server")
                            error_count += 1
                            continue

                    ## cannot collect the data on second loop
                    if error_count > 50:
                        trouble_file.append(file)
                        break

                    success_flag2 = 0
                    while success_flag2 == 0 and error_count <= 50:
                        try:
                            response = json_gpt(query)
                            response["Context"] = Emr_split_data[i]
                            context_emb = get_embedding(response["Context"])
                            explanation_emb = get_embedding(response["Explanation"])
                            cosine_sim = (dot(context_emb, explanation_emb) + 1) / 2
                            response["confidence"] = cosine_sim
                            response["text_file"] = file
                            response["sentence_index"] = i
                            temp2.append(response)
                            success_flag2 = 1
                            #print(response)

                        except:
                            time.sleep(10)
                            print("error occured in chatgpt server")
                            continue

                    ## cannot collect the data on second loop
                    if error_count > 50:
                        trouble_file.append(file)
                        break
                ### if every setence in paragraph didn't occur error
                ### Save as dataset

                dataset.extend(temp)
                dataset1.extend(temp1)
                dataset2.extend(temp2)

        folder_deque.popleft()

    end = time.time()

    print("total_time:", end-start)

    with open('test_multi_label_dataset_part4_v1.json', 'w') as f:
        ## each json part has the 100 folders of dataset.
        json.dump(dataset, f)

    with open('test_multi_label_dataset_part4_v2.json', 'w') as f:
        ## each json part has the 100 folders of dataset.
        json.dump(dataset1, f)

    with open('test_multi_label_dataset_part4_v3.json', 'w') as f:
        ## each json part has the 100 folders of dataset.
        json.dump(dataset2, f)

    remain_folders = np.array(folder_deque)
    np.save('remain_folders_test.npy', remain_folders)
    np.save('trouble_para_test.npy', trouble_file)

