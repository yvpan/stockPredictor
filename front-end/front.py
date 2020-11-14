import xlrd

data = xlrd.open_workbook('./database.xlsx')
sheet = data.sheet_by_name('Sheet1')
write = open('frontOut.txt', 'w')
rows = sheet.nrows
data_value = []
for i in range(rows):
    data_value.append(sheet.row_values(i, start_colx=0, end_colx=2))
copy = data_value.copy()
one_dim = [n.lower() for a in data_value for n in a]
company = [a[1].lower().split(' ') for a in data_value]

inter_kind = eval(input('Please choose enter stock symbol or company name, 1.symbol  2.company name:'))
if inter_kind == 1:
    inter = input('Please type a stock symbol：').split(' ')[0].lower()
    if inter in one_dim:
        for i in range(rows):
            if copy[i][0].lower() == inter:
                write.write(data_value[i][0])
                write.write('\t')
                write.write(data_value[i][1])
                write.write('\n')
                break
    else:
        print('The stock you search for does not exist')
elif inter_kind == 2:
    inter = input('Please type a company name：').lower()
    flag = 0
    for i in range(rows):
        if inter in company[i]:
            write.write(data_value[i][0])
            write.write('\t')
            write.write(data_value[i][1])
            write.write('\n')
            flag = 1
            break
    if flag == 0:
        print('the company is not in the list')
else:
    print('input error')
write.close()
