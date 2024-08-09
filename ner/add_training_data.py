# import json
# import os

# def load_train_data(json_path):
#     if os.path.exists(json_path):
#         with open(json_path, 'r', encoding='utf-8') as f:
#             train_data = json.load(f)
#     else:
#         train_data = []
#     return train_data

# def save_train_data(train_data, json_path):
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(train_data, f, ensure_ascii=False, indent=4)

# def add_training_example(train_data):
#     print("Enter training example:")
#     text = input("Text: ")
    
#     entities = []
#     while True:
#         start = input("Entity start (or type 'done' to finish): ")
#         if start.lower() == 'done':
#             break
#         start = int(start)
#         end = int(input("Entity end: "))
#         label = input("Entity label: ")
#         entities.append([start, end, label])
    
#     annotations = {"entities": entities}
#     train_data.append((text, annotations))

# def main():
#     train_data_path = 'train_data.json'
#     train_data = load_train_data(train_data_path)
    
#     while True:
#         add_training_example(train_data)
#         cont = input("Add another example? (yes/no): ")
#         if cont.lower() != 'yes':
#             break
    
#     save_train_data(train_data, train_data_path)
#     print(f"Training data saved to {train_data_path}")

# if __name__ == "__main__":
#     main()
