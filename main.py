# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
import FirstModel
import util
import TrimmingData as TrimmingData
from flask import Flask, request
import json

path = './1stmodelstate.h5'
model_params = torch.load(path,map_location=torch.device('cpu'))
model = FirstModel.BaseModel()
model.load_state_dict(model_params)

consultant_data_name = "consultant" 
student_data_name = "student"
consultant_total_data = util.data_init(consultant_data_name)
student_total_data = util.data_init(student_data_name)
DC = TrimmingData.DataComparison()
seed = 201900278
np.random.seed(seed)
app = Flask(__name__)


@app.route("/", methods=['GET'])
def initialization():
    if request.method == 'GET':

        show_num = int(request.args.get('show',100))
        page = int(request.args.get('page',1))
        user_type = request.args.get('user_type','student')

        if user_type=='student':
            total_data = student_total_data
        else:
            total_data = consultant_total_data

        user_data = request.get_json()
        user_df = pd.DataFrame([user_data],index=[0])
        user_df = util.Kor2Eng(user_df)

        # 상담자 데이터를 같은 그룹인 데이터를 뽑아낸후 같은 직업을 앞에오게끔 구성
        ordered_user_df = util.find_user_group(user_info=user_df, data=total_data)
        
        if len(ordered_user_df)==0:
            return "No candidate"
        
        start = (page-1)*show_num
        end = page*show_num
        if len(ordered_user_df) < start :
            return "Over Size"

        if len(ordered_user_df) < end:
            end = len(ordered_user_df)-1
            
        first_candidate_df = ordered_user_df.loc[start:end-1] # end였는데 end-1로 바꿈
        first_candidate_df.reset_index(drop=True,inplace=True)

        score_list = []

        input_data = torch.FloatTensor(DC.prepare_data(user_df,first_candidate_df).values)
        score = model(input_data)
        
        score = score.view(-1).detach().numpy()
        first_candidate_df['score'] = score

        return_df = first_candidate_df.sort_values(by='score',ascending = False)
        return_df = util.Eng2Kor(return_df)
        result = []
        for index, row in return_df.iterrows():
            temp = row.to_dict()
            result.append(temp)
        
        result_json = json.dumps(result)
        return result_json

    return "successfully"

@app.route("/update",methods=['GET'])
def update():
    if request.method == 'GET':
        # 진성이가 index +9000해서 보내줌
        # 회원가입할떄 수정됨
        update_user_data = request.get_json()
        update_user_df = pd.DataFrame([update_user_data])
        update_user_df = util.Kor2Eng(update_user_df)        
        return_data_tup = util.make_data(update_user_df)
        DC.update_dict(return_data_tup)
    return "successfully"
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug = True)

