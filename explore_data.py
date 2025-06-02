import os

# Dynamically calculate the path to data/landmarks, no matter where it's called from
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "landmarks"))

def explore_dataset():
    print("Welcome to the landsmark ML Pipeline!")
    print("First: Exploring the dataset we have:")

    try:
        for class_name in os.listdir(base_path):
            class_path = os.path.join(base_path, class_name)
            if os.path.isdir(class_path):
                imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"{class_name}: {len(imgs)} images")
    except FileNotFoundError:
        print(f" Directory not found: {base_path}")

# Only execute if run directly
if __name__ == "__main__":
    explore_dataset()
