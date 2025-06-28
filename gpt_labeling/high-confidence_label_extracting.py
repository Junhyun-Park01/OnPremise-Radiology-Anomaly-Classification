import json

### load 3 dataset file (independant 3 trial for GPT)
file1 = open('multi_label_dataset_final_v1.json')
file2 = open('multi_label_dataset_final_v2.json')
file3 = open('multi_label_dataset_final_v3.json')

file1 = list(json.load(file1))
file2 = list(json.load(file2))
file3 = list(json.load(file3))

final_dict = []
count = 0
count1 = 0
count2 = 0

for i in range(len(file1)):
    label_list = []
    score_list = []
    sentence_dict = {}
    if (file1[i]['Result'] == file2[i]['Result'] == file3[i]['Result']):
        sentence_dict['Result'] = file1[i]['Result']
        sentence_dict['Context'] = file1[i]['Context']
        sentence_dict['text_file'] = file1[i]['text_file']
        sentence_dict['sentence_index'] = file1[i]['sentence_index']
        sentence_dict['confidence'] = (file1[i]['confidence'] + file2[i]['confidence'] + file3[i]['confidence']) / 3
        # print("######################## All collect ################")

    else:
        label_list.append(file1[i]['Result'])
        label_list.append(file2[i]['Result'])
        label_list.append(file3[i]['Result'])

        score_list.append(file1[i]['confidence'])
        score_list.append(file2[i]['confidence'])
        score_list.append(file3[i]['confidence'])

        count += 1

        if len(set(label_list)) == 3:
            count1 += 1

        else:
            label_set = list(set(label_list))
            elem1_count = 0
            elem1_score = 0
            elem2_score = 0
            elem2_count = 0

            for x in range(len(label_list)):

                if(label_list[x] == label_set[0]):
                    elem1_count += 1
                    elem1_score += score_list[x]

                elif (label_list[x] == label_set[1]):
                    elem2_count += 1
                    elem2_score += score_list[x]

            elem1_score = elem1_score / elem1_count
            elem2_score = elem2_score / elem2_count

            if elem1_count > elem2_count and elem1_score > elem2_score:
                count2 += 1
                # print("")
                # print(file1[i])
                # print(file2[i])
                # print(file3[i])
                #

                sentence_dict['Result'] = label_set[0]
                sentence_dict['Context'] = file1[i]['Context']
                sentence_dict['text_file'] = file1[i]['text_file']
                sentence_dict['sentence_index'] = file1[i]['sentence_index']
                sentence_dict['confidence'] = elem1_score

            elif elem1_count < elem2_count and elem1_score < elem2_score:
                count2 += 1
                # print("")
                # print(file1[i])
                # print(file2[i])
                # print(file3[i])

                sentence_dict['Result'] = label_set[1]
                sentence_dict['Context'] = file1[i]['Context']
                sentence_dict['text_file'] = file1[i]['text_file']
                sentence_dict['sentence_index'] = file1[i]['sentence_index']
                sentence_dict['confidence'] = elem2_score


    # print(sentence_dict)
    # print(len(sentence_dict))
    if len(sentence_dict) != 0:
        final_dict.append(sentence_dict)

print("total length of sentence", len(file1))
print("sentence that doesn't have all same lables", count)  ### sentence that has some issues
print("sentence that has three different labels",count1) ### sentence that has all different labels
print("majority of labels and bigger confidence",count2) ### Sentence that can solve the problem
print("that can use as data", count - count1 - count2)


file1.extend(file2)
file1.extend(file3)

print(file1)
print(len(final_dict))
out_file = open("multi_label_dataset_final.json", "w")
json.dump(final_dict, out_file)
out_file.close()
