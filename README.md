# BF-PSR-Framework

Reproducibility: for reproducibility, all codes, detailed results, parameter values tested, and data studied in our paper: “How to take advantage of behavioral features for the early detection of grooming in online conversations”, including PJZ and PJZC datasets, are freely available for download in this repository.

## Folders
- Concurrent Methods: Contains the state-of-the-art methods of the preventive grooming detection area, i.e., the ENB, the PSR, the three variants of method MulR, and our proposal, the BF-PSR framework. Please note that some files were compressed due to Github's space limit.
- Data: This folder contains the data used by the state-of-the-art methods: From the SGD training and testing sets to the new proposed PJZ and PJZC datasets. Please refer to the original version of the SGD corpus in: https://pan.webis.de/publications.html?q=bibid%3Ainches_2012 
  Besides, SGD, PJZ, and PJZC datasets are already adapted to be used by the BF-PSR  framework, i.e., they contain the time when a conversation starts, intervention words per user, and the number of participants in a conversation behavioral features.  	
- JSON data: To obtain the PJZ and PJZC datasets in JSON format please compile the conversations_to_json.py file. It contains the conversations in a JSON structure without the behavioral features.

## PJZ and PJZC datasets Json format
To mitigate the shortage of data in the area of online grooming detection are assembled and studied two new datasets, which we named as PJZ and PJZC. 
The Json format of the datasets contains the flow of the conversations in a practical format:
- id: identifier of the conversation
- source: Source of the conversation, i.e., PJ chats, #ZIG channel chats, Chit chat data.
- label: (0) non-groomer, (1) groomer.
- messages: List of messages with author, time and text atributes.

```json
{
            "id": 0,
            "source": "PJ chats",
            "label": "1",
            "messages": [
                {
                    "author": "decoy",
                    "time": "14:40",
                    "text": "Hey Its Mads"
                },
                {
                    "author": "Billy Joe",
                    "time": "14:40",
                    "text": "Hey babes"
                },
                {
                    "author": "Billy Joe",
                    "time": "14:41",
                    "text": "Almost done with work. So glad"
                },
                {
                    "author": "decoy",
                    "time": "14:41",
                    "text": "Hey:) that's good right?"
                },
                {
                    "author": "Billy Joe",
                    "time": "14:42",
                    "text": "Yea"
                },
                
                ...
            ]
        }
```

## Usage

To extract the conversations from the Json structure follow the following Python example.

```python
import json
with open('PJZ.txt') as json_file:
    conversations = json.load(json_file)
    for p in conversations['conversation']:
        m_id = 1
        print('Source: ' + p['source'])
        print('Label: ' + p['label'])
        for message in p['messages']:
            print(' Message: ',m_id)
            print("     Author: " + message['author'])
            print("     Text: " + message['text'])
            print("     Time: " + message['time'])
            m_id += 1
        print('*'*20)
