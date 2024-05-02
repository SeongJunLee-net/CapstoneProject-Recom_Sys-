# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import os
import torch
import torch.nn.functional as F


class DataComparison():
    path = "./"    
    with open(path+'big_company_kind_vec_dict.pkl','rb') as f:
        big_company_kind_vec_dict = pickle.load(f)
    with open(path+'mid_company_kind_vec_dict.pkl','rb') as f:
        mid_company_kind_vec_dict = pickle.load(f)
    with open(path+'company_kind_vec_dict.pkl','rb') as f:
        company_kind_vec_dict = pickle.load(f)
    with open(path+'job_vec_dict.pkl','rb') as f:
        job_vec_dict = pickle.load(f)
    with open(path+'field_vec_dict.pkl','rb') as f:
        field_vec_dict = pickle.load(f)
    with open(path+'major_vec_dict.pkl','rb') as f:
        major_vec_dict = pickle.load(f)
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def cat_data_matching(self, elem1 : pd.Series, elem2 : pd.Series):
        #print(111) # 반응
        return (elem1==elem2)
    def make_similarity(self,keyword : str, user_index : int, candidate_index : pd.Series):
        if keyword == "big_company_kind":
            dictionary = self.big_company_kind_vec_dict
        elif keyword == "mid_company_kind":
            dictionary = self.mid_company_kind_vec_dict
        elif keyword == "company_kind":
            dictionary = self.company_kind_vec_dict
        elif keyword == "job":
            dictionary = self.job_vec_dict
        elif keyword == "field":
            dictionary = self.field_vec_dict
        elif keyword == "major":
            dictionary = self.major_vec_dict

        candidate_idx_list = [idx for idx in candidate_index]
        
        user_tensor = torch.Tensor().to(self.device)
        candidate_tensor = torch.Tensor().to(self.device)
        
        tmp = dictionary[user_index]
        user_tensor = tmp.repeat(len(candidate_index),1)
        for idx in candidate_index:
            tmp = dictionary[idx]
            candidate_tensor = torch.concat([candidate_tensor,tmp],dim=0)

        cos_sim_vec = F.cosine_similarity(user_tensor, candidate_tensor, dim=1)
        cos_sim_vec = cos_sim_vec.to('cpu').numpy()
        return cos_sim_vec
    
    def prepare_data(self, user : pd.DataFrame, candidate : pd.DataFrame):
        user_index = user['index'].item()
        user = pd.concat([user]*len(candidate), ignore_index=True)
        matching_dict = {
            'mat_sex':self.cat_data_matching(user['sex'],candidate['sex']),
            'mat_company_name':self.cat_data_matching(user['company_name'],candidate['company_name']),
            'mat_company_scale':self.cat_data_matching(user['company_scale'],candidate['company_scale']),
            'mat_univ_kind':self.cat_data_matching(user['univ_kind'],candidate['univ_kind']),
            'mat_univ_loc':self.cat_data_matching(user['univ_loc'],candidate['univ_loc']),
            'mat_univ_name':self.cat_data_matching(user['univ_name'],candidate['univ_name']),
            'mat_univ_main_branch':self.cat_data_matching(user['univ_main_branch'],candidate['univ_main_branch']),
            'mat_univ_day_night':self.cat_data_matching(user['univ_day_night'],candidate['univ_day_night']),
            'mat_education':self.cat_data_matching(user['education'],candidate['education']),
#             'mat_group':self.cat_data_matching(data['group'],data['student_group'])
        }
 
        simil_dict = {
            'simil_company_kind':self.make_similarity("company_kind",user_index,candidate['index']),
            'simil_mid_company_kind':self.make_similarity("mid_company_kind",user_index,candidate['index']),
            'simil_big_company_kind':self.make_similarity("big_company_kind",user_index,candidate['index']),
            'simil_major':self.make_similarity("major",user_index,candidate['index']),
            'simil_job':self.make_similarity("job",user_index,candidate['index']),
            'simil_field':self.make_similarity("field",user_index,candidate['index'])
        }

        result_dict = {}
        result_dict.update(matching_dict)
        result_dict.update(simil_dict)
            
        result = pd.DataFrame(result_dict)
    
        result = result.astype(float)
        return result
    
    def update_dict(cls,dict_tup:tuple):
        path = cls.path
        cls.big_company_kind_vec_dict.update(dict_tup[0])
        cls.mid_company_kind_vec_dict.update(dict_tup[1]) 
        cls.company_kind_vec_dict.update(dict_tup[2])
        cls.major_vec_dict.update(dict_tup[3])
        cls.job_vec_dict.update(dict_tup[4])
        cls.field_vec_dict.update(dict_tup[5])

    

    # def update_dict(cls,dict_tup:tuple):
    #     path = cls.path
    #     with open(path+'check_update.pkl','rb') as f:
    #         big_company_kind_dict = pickle.load(f)
    #     with open(path+'check_update.pkl','wb') as f:
    #         big_company_kind_dict.update(dict_tup[0])
    #         pickle.dump(big_company_kind_dict,f)

    #     with open(path+'check_update.pkl','rb') as f:
    #         mid_company_kind_vec_dict = pickle.load(f)
    #     with open(path+'check_update.pkl','wb') as f:
    #         mid_company_kind_vec_dict.update(dict_tup[1])
    #         pickle.dump(mid_company_kind_vec_dict,f)

    #     with open(path+'check_update.pkl','rb') as f:
    #         company_kind_vec_dict = pickle.load(f)
    #     with open(path+'check_update.pkl','wb') as f:
    #         company_kind_vec_dict.update(dict_tup[2])
    #         pickle.dump(company_kind_vec_dict,f)

    #     with open(path+'check_update.pkl','rb') as f:
    #         major_vec_dict = pickle.load(f)
    #     with open(path+'check_update.pkl','wb') as f:
    #         major_vec_dict.update(dict_tup[3])
    #         major_vec_dict = pickle.dump(major_vec_dict,f)

    #     with open(path+'check_update.pkl','rb') as f:
    #         job_vec_dict = pickle.load(f)
    #     with open(path+'check_update.pkl','wb') as f:
    #         job_vec_dict.update(dict_tup[4])
    #         job_vec_dict = pickle.dump(job_vec_dict,f)
        
    #     with open(path+'check_update.pkl','rb') as f:
    #         field_vec_dict = pickle.load(f)
    #     with open(path+'check_update.pkl','wb') as f:
    #         field_vec_dict.update(dict_tup[5])
    #         field_vec_dict = pickle.dump(field_vec_dict,f)
        

