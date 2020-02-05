# create special case text file

f_special_case = open("transfered_special_case.txt","w")

with open("special_case.txt","r") as f:
    for line in f:
        transfered_special_case_example = line.split("/")[0].replace('"','').replace(':','').replace(',','')
        f_special_case.write(transfered_special_case_example+"\n") 
        #just use the most common case

f_special_case.close()

f_special_case = open("transfered_special_case.txt","r")

with open('transfered_special_case_2.txt', 'w') as f_special_case_2:
    for line in f_special_case:
        if not line.strip():
            continue  # skip the empty line
        line_split = line.split(" ")
        capitalized_line = ""
        for l in range(len(line_split)):
            if l == 0 or l == 1:
                capitalized_line += line_split[l].capitalize()
            else:
                capitalized_line += line_split[l]
            if l < len(line_split)-1:
                capitalized_line += " "
        f_special_case_2.write(line)  # non-empty line. Write it to output
        f_special_case_2.write(capitalized_line)

f_special_case.close()