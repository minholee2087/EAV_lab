from main_classwise import *
import os

# Please make sure to download full dataset from the following link:
# https://zenodo.org/records/10205702

# for 1 subject example run go with these links to get 'dataset_processing/', 'pretrained_models/', 'Finetuned_models/':
# https://drive.google.com/drive/folders/13tGH7TJEtokCIZo1hQF0MgueHSN3fqwa?usp=sharing

def download_data():
    print("\n⏳ Downloading data from Google Drive...")
    os.system("gdown --folder https://drive.google.com/drive/folders/13tGH7TJEtokCIZo1hQF0MgueHSN3fqwa?usp=sharing")
    print("\n✅ Data download complete!")

def run_all_classes():
    print("\n🚀 Running All Classes Mode...")
    train_all_classes()

def run_zeroshot():
    print("\n🤖 Running Zero-Shot Mode...")
    print("\n There are 5 emotion classes labeled as follows:")
    print("\n Neutral = 0 \n Sadness = 1 \n Anger = 2 \n Happiness = 3 \n Calmness = 4 \n ")
    choice = input("👉 Enter class label from 0 to 4: ").strip()
    if choice == "0":
        train_zero_shot(0)
    elif choice == "1":
        train_zero_shot(1)
    elif choice == "2":
        train_zero_shot(2)
    elif choice == "3":
        train_zero_shot(3)
    elif choice == "4":
        train_zero_shot(4)
    else:
        print("\n⚠️ Invalid input! Please restart and enter class label from 0 to 4.")


if __name__ == "__main__":
    print("\n🌟 Welcome to the Temporal Repository. Here you can run an example of a chosen code. 🌟")

    download_choice = input("\n📥 Do you need to download data from Google Drive? (y/n): ").strip().lower()
    if download_choice == "y":
        download_data()

    print("\n📂 Please ensure the downloaded data is placed in this repository's folder, following the correct folder structure.")

    print("\nPlease choose how you want to run the program:\n")
    print("1️⃣  All Classes - Train and test on all available classes (5 emotions)")
    print("2️⃣  Zero-Shot - Test on one unseen class (4 emotions)\n")

    choice = input("👉 Enter your choice (1 or 2): ").strip()

    if choice == "1":
        run_all_classes()
    elif choice == "2":
        run_zeroshot()
    else:
        print("\n⚠️ Invalid input! Please restart and enter '1' or '2'.")