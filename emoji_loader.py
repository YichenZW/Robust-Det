import json

class emoji_list(object):
    def __init__(self, json_dir="emoji.json"):
        with open(json_dir, 'r') as file:
            self.all_dict = json.load(file)
        self.list = [item['emoji'] for item in self.all_dict if 'emoji' in item.keys()]
