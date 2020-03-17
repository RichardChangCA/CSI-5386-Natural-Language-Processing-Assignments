file_name_1 = "results_part_1.txt"
file_name_2 = "results_part_1_stacked_lstm.txt"
file_name_3 = "results_part_1_bert.txt"

file_list = [file_name_1,file_name_2,file_name_3]

def calculate_precision(TP,TN,FP,FN):
    if TP+FP == 0:
        return 0
    else:
        return TP/(TP+FP)

def calculate_recall(TP,TN,FP,FN):
    if TP+FN == 0:
        return 0
    else:
        return TP/(TP+FN)

def calculate_f_measure(TP,TN,FP,FN):
    TP = int(TP)
    TN = int(TN)
    FP = int(FP)
    FN = int(FN)
    precision = calculate_precision(TP,TN,FP,FN)
    recall = calculate_recall(TP,TN,FP,FN)
    if precision+recall == 0:
        f_measure = 0
    else:
        f_measure = (2*precision*recall)/(precision+recall)
    return precision,recall,f_measure

new_results = open("new_result.txt",'w')

for i in file_list:
    new_results.write('\n\n' + i + '\n\n')
    with open(i,'r') as f:
        content = f.readlines()
        entailment = []
        neutral = []
        contraction = []
        for n in range(len(content)):
            if n == 0:
                continue
            elif n >= 1 and n <= 4:
                entailment.append(content[n].split(':')[1][1:-1])
            elif n >= 5 and n <= 8:
                neutral.append(content[n].split(':')[1][1:-1])
            elif n >= 9 and n <= 11:
                contraction.append(content[n].split(':')[1][1:-1])
            else: #n=12
                contraction.append(content[n].split(':')[1][1:])
        classes = [entailment,neutral,contraction]
        for c in classes:
            if c == entailment:
                new_results.write("entailment:\n")
            elif c == neutral:
                new_results.write("neutral:\n")
            else:
                new_results.write("contraction:\n")
            precision,recall,f_measure = calculate_f_measure(c[0],c[1],c[2],c[3])
            new_results.write("precision:"+str(precision)+"\n")
            new_results.write("recall:"+str(recall)+"\n")
            new_results.write("f_measure:"+str(f_measure)+"\n")

new_results.close()

        
