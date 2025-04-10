import pandas as pd


def get_info_dataset(filepath):

    dataset = pd.read_csv(filepath)
    TCR_list = list(dataset['CDR3'].dropna())
    antigen_list = list(dataset['Antigen'])
    HLA_list = list(dataset['HLA'])

    HLA_TCR_antigen=[]
    HLA_antigen=[]
    for i in range(len(TCR_list)):
        HLA_TCR_antigen.append(TCR_list[i]+'_'+antigen_list[i]+'_'+HLA_list[i])
        HLA_antigen.append(antigen_list[i]+'_'+HLA_list[i])

    return list(set(HLA_TCR_antigen)),list(set(HLA_antigen))


def find_intersection_and_duplicates(list1, list2):
    # 将列表转换为集合
    set1 = set(list1)
    set2 = set(list2)

    # 求交集
    intersection = set1 & set2

    # 如果交集非空，则打印交集中的元素
    if intersection:
        print("交集中的元素:", intersection)
    else:
        print("没有交集")

    # 打印出重复的元素
    duplicates = set()
    for item in list1:
        if item in set2:
            duplicates.add(item)

    if duplicates:
        print("重复的元素:", duplicates)
    else:
        print("没有重复的元素")
    return duplicates


HLA_TCR_antigen1,HLA_antigen1=get_info_dataset('data/prolymphocyticleukemia_pHLA_cdr3_umis_barcode.csv')
HLA_TCR_antigen2,HLA_antigen2=get_info_dataset('data/del_positive sample_donor1-3_iedb_vdjdb.csv')

print("HLA_TCR_antigen1：",len(HLA_TCR_antigen1))
print("HLA_TCR_antigen2：",len(HLA_TCR_antigen2))
print("HLA_antigen1：",len(HLA_antigen1))
print("HLA_antigen2：",len(HLA_antigen2))


duplicates1=find_intersection_and_duplicates(HLA_TCR_antigen1,HLA_TCR_antigen2)
print("HLA_TCR_antigen重复：",len(duplicates1))
duplicates2=find_intersection_and_duplicates(HLA_antigen1,HLA_antigen2)
print("HLA_antigen重复：",len(duplicates2))