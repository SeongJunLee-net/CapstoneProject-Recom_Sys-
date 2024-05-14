import pandas as pd
import numpy as np
import copy
import os
import torch
import pickle
from transformers import RobertaModel, RobertaTokenizer

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
roberta = RobertaModel.from_pretrained(model_name).to(device)
seed = 201900278

path = '.'
with open(path+'/Kor2EngPickle/big_company_dict.pkl','rb') as f:
    big_company_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/mid_company_dict.pkl','rb') as f:
    mid_company_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/company_kind_dict.pickle','rb') as f:
    company_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/company_scale_dict.pickle','rb') as f:
    company_scale_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/education_dict.pickle','rb') as f:
    education_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/field_dict.pickle','rb') as f:
    field_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/job_dict.pickle','rb') as f:
    job_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/major_dict_revise.pickle','rb') as f:
    major_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/univ_day_night_dict.pickle','rb') as f:
    univ_day_night_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/univ_kind_dict.pickle','rb') as f:
    univ_kind_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/univ_loc_dict.pickle','rb') as f:
    univ_loc_trans = pickle.load(f)
with open(path+'/Kor2EngPickle/univ_main_branch_dict.pickle','rb') as f:
    univ_main_branch_trans = pickle.load(f)


with open(path+'/Eng2KorPickle/big_company_dict_reverse.pkl','rb') as f:
    big_company_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/mid_company_dict_reverse.pkl','rb') as f:
    mid_company_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/company_kind_dict_reverse.pkl','rb') as f:
    company_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/company_scale_dict_reverse.pkl','rb') as f:
    company_scale_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/education_dict_reverse.pkl','rb') as f:
    education_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/field_dict_reverse.pkl','rb') as f:
    field_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/job_dict_reverse.pkl','rb') as f:
    job_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/major_dict_revise_reverse.pkl','rb') as f:
    major_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/univ_day_night_dict_reverse.pkl','rb') as f:
    univ_day_night_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/univ_kind_dict_reverse.pkl','rb') as f:
    univ_kind_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/univ_loc_dict_reverse.pkl','rb') as f:
    univ_loc_trans_E = pickle.load(f)
with open(path+'/Eng2KorPickle/univ_main_branch_dict_reverse.pkl','rb') as f:
    univ_main_branch_trans_E = pickle.load(f)



def data_init(data_name:str):
    path = './'
    data = pd.read_csv(path+f'{data_name}.csv')
    return data

def find_user_group(user_info:pd.DataFrame, data : pd.DataFrame,
                     univ_rate:int,company_rate:int,job_rate:int):    
    np.random.seed(seed)
    # total_rate = 3
    # user_job = user_info['job'].item()
    user_company_name = user_info['company_name'].item()
    user_company_kind = user_info['company_kind'].item() 
    user_univ_name = user_info['univ_name'].item()
    user_major = user_info['major'].item()
    user_group = user_info['group'].item() 
    
    group_condition = (data['group'] == user_group)
    univ_condition = (data['univ_name'] == user_univ_name)
    company_name_condition = (data['company_name'] == user_company_name)
    major_condition = (data['major']==user_major)
    company_kind_condition = (data['company_kind']==user_company_kind)


    
    group_data = copy.deepcopy(data.loc[group_condition])
    #return_group_data = group_data.sample(frac = job_rate/total_rate)
    
    univ_data = copy.deepcopy(data.loc[univ_condition])
    #return_univ_data = univ_data.sample(frac = univ_rate/total_rate)
    
    company_data = copy.deepcopy(data.loc[company_name_condition])
    #return_company_data = company_data.sample(frac = company_rate/total_rate)
    
    major_data = copy.deepcopy(data.loc[major_condition])

    company_kind_data = copy.deepcopy(data.loc[company_kind_condition])

    candidate_data = [(company_rate,company_kind_data),(univ_rate,major_data),
                      (job_rate,group_data),(company_rate,company_data),
                      (univ_rate,univ_data)]
    candidate_data.sort(key=lambda x:x[0],reverse=True)
    candidate_data = [df[1] for df in candidate_data]

    #return_data = pd.concat([return_group_data,return_univ_data,return_company_data],axis=0,ignore_index=True)
    #return_data.drop_duplicates(inplace=True)

    #drop_idx = return_data.index
    candidate_data = pd.concat(candidate_data,axis=0,ignore_index=True)
    candidate_data.drop_duplicates(inplace=True)
    #candidate_data.drop(index=drop_idx,inplace=True)

    #return_data = return_data.sample(frac=1)
    #return_data = pd.concat([return_data,candidate_data],axis=0,ignore_index=True)
    ## return_job_data = copy.deepcopy(return_group_data.loc[job_condition])
    ## return_group_data.drop(index=return_job_data.index, inplace = True)
    ## return_data = pd.concat([return_job_data,return_group_data],axis=0,ignore_index=True)
    return_data = candidate_data
    return return_data


def make_vector(keyword : str, elem : str):
    if keyword == "company":
        prompt = "My {} does {} business."
    elif keyword == "major":
        prompt = "My {} is {}."
    elif keyword == "field":
        prompt = "My {} of major is {}."
    elif keyword == "job":
        prompt = "My {} is {}."

    vector_str = prompt.format(keyword,elem)
    tokenizer_output = tokenizer.tokenize(vector_str)
    
    model_input = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenizer_output]
    
    model_input = torch.tensor(model_input).unsqueeze(0).to(device)
    with torch.no_grad():
        roberta.eval()
        model_output = torch.mean(roberta(model_input)[0],dim=1)
    return model_output



def Eng2Kor(update_data : pd.DataFrame):
    for cnt,data in update_data.iterrows():
        update_data.loc[cnt,'big_company_kind'] = big_company_trans_E[update_data.loc[cnt,'big_company_kind']]
        update_data.loc[cnt,'mid_company_kind'] = mid_company_trans_E[update_data.loc[cnt,'mid_company_kind']]
        update_data.loc[cnt,'company_kind'] = company_trans_E[update_data.loc[cnt,'company_kind']]
        update_data.loc[cnt,'company_scale'] = company_scale_trans_E[update_data.loc[cnt,'company_scale']]
        update_data.loc[cnt,'education'] = education_trans_E[update_data.loc[cnt,'education']]
        update_data.loc[cnt,'field'] = field_trans_E[update_data.loc[cnt,'field']]
        update_data.loc[cnt,'job'] = job_trans_E[update_data.loc[cnt,'job']]
        update_data.loc[cnt,'major'] = major_trans_E[update_data.loc[cnt,'major']]
        update_data.loc[cnt,'univ_day_night'] = univ_day_night_trans_E[update_data.loc[cnt,'univ_day_night']]
        update_data.loc[cnt,'univ_kind'] = univ_kind_trans_E[update_data.loc[cnt,'univ_kind']]
        update_data.loc[cnt,'univ_loc'] = univ_loc_trans_E[update_data.loc[cnt,'univ_loc']]
        update_data.loc[cnt,'univ_main_branch'] = univ_main_branch_trans_E[update_data.loc[cnt,'univ_main_branch']]
    return update_data

def Kor2Eng(update_data : pd.DataFrame):
    for cnt,data in update_data.iterrows():
        update_data.loc[cnt,'big_company_kind'] = big_company_trans[update_data.loc[cnt,'big_company_kind']]
        update_data.loc[cnt,'mid_company_kind'] = mid_company_trans[update_data.loc[cnt,'mid_company_kind']]
        update_data.loc[cnt,'company_kind'] = company_trans[update_data.loc[cnt,'company_kind']]
        update_data.loc[cnt,'company_scale'] = company_scale_trans[update_data.loc[cnt,'company_scale']]
        update_data.loc[cnt,'education'] = education_trans[update_data.loc[cnt,'education']]
        update_data.loc[cnt,'field'] = field_trans[update_data.loc[cnt,'field']]
        update_data.loc[cnt,'job'] = job_trans[update_data.loc[cnt,'job']]
        update_data.loc[cnt,'major'] = major_trans[update_data.loc[cnt,'major']]
        update_data.loc[cnt,'univ_day_night'] = univ_day_night_trans[update_data.loc[cnt,'univ_day_night']]
        update_data.loc[cnt,'univ_kind'] = univ_kind_trans[update_data.loc[cnt,'univ_kind']]
        update_data.loc[cnt,'univ_loc'] = univ_loc_trans[update_data.loc[cnt,'univ_loc']]
        update_data.loc[cnt,'univ_main_branch'] = univ_main_branch_trans[update_data.loc[cnt,'univ_main_branch']]
    return update_data

def make_data(update_data : pd.DataFrame):
    big_company_kind_vec_dict = {}
    mid_company_kind_vec_dict = {}
    company_kind_vec_dict = {}
    major_vec_dict = {}
    job_vec_dict = {}
    field_vec_dict = {}
    for idx,data in update_data.iterrows():
        data = data.to_frame().T
        vec_dict = {
            'company_kind_vec':make_vector("company",data['company_kind'].item()),
            'mid_company_kind_vec':make_vector("company",data['mid_company_kind'].item()),
            'big_company_kind_vec':make_vector("company",data['big_company_kind'].item()),
            'major_vec':make_vector("major",data['major'].item()),
            'job_vec':make_vector("job",data['job'].item()),
            'field_vec':make_vector("field",data['field'].item())
        }
        user_id = data['index'].item()
        big_company_kind_vec_dict[user_id] = vec_dict['big_company_kind_vec'].detach().to('cpu')
        mid_company_kind_vec_dict[user_id] = vec_dict['mid_company_kind_vec'].detach().to('cpu')
        company_kind_vec_dict[user_id] = vec_dict['company_kind_vec'].detach().to('cpu')
        major_vec_dict[user_id] = vec_dict['major_vec'].detach().to('cpu')
        job_vec_dict[user_id] = vec_dict['job_vec'].detach().to('cpu')
        field_vec_dict[user_id] = vec_dict['field_vec'].detach().to('cpu')
    return (big_company_kind_vec_dict,
            mid_company_kind_vec_dict,
            company_kind_vec_dict,
            major_vec_dict,
            job_vec_dict,
            field_vec_dict)
