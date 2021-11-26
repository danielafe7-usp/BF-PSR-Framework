import json
import pickle

def saving_pkl(file,name):

    with open(name,'wb') as f:
            pickle.dump(file,f)
            
def loading_pkl(name):
    with open(name, 'rb') as f:
        file = pickle.load(f)
    return file


def reading_conversations(PATH_list,name):
    '''Json format '''
    conversations_json = {}
    conversations_json['conversation'] = []
    num_conversations = 0

    for path,source,label in PATH_list:
        id = 0
        
        conversations = loading_pkl(path)
        for groomer_id,sub_conversations in conversations[:]:
            #print(id,groomer_id,len(sub_conversations))
            for sub_c in sub_conversations:
                messages_l = []
                for message in sub_c:
                    author = message[0]
                    time = message[1]
                    text = message[2]

                    messages_l.append(
                        {
                            "author":author,
                            "time": time,
                            "text": text,
                        })
                    
                '''Saving conversation into Json format'''
                conversations_json['conversation'].append({
                'id':num_conversations,
                'source': source,
                'label': label,
                "messages": messages_l })

                num_conversations += 1
            id += 1
    
    print("Json structure ...")
    print(json.dumps(conversations_json, indent=4))
    print("Num_conversations: ",num_conversations)
    with open(name+'.txt', 'w') as outfile:
        json.dump(conversations_json, outfile)
    
    
'''Init parameters'''
name = None   
source_1 = "PJ chats"
source_2 = "ZIG chats"
source_3 = "Chit chats"

groomer = '1'
non_groomer = '0'

zig_conversations =  "Conversations/ZIG.pkl"
chit_conversations = "Conversations/Chit.pkl"
pj_conversations =   "Conversations/PJ.pkl"


value = input("Please enter 0/1 to generate the PJZ/PJZC dataset\n")
if value == '0':
    print("Generating PJZ dataset ...")
    name = 'PJZ'
    reading_conversations([(pj_conversations,source_1,groomer),(zig_conversations,source_2,non_groomer)],name)
elif value == '1':
    print("Generating PJZC dataset ...")
    name = 'PJZC'
    reading_conversations([(pj_conversations,source_1,groomer),(zig_conversations,source_2,non_groomer),(chit_conversations,source_3,non_groomer)],name)
else:
    print("You  must choose betwenn 0/1 ...")



