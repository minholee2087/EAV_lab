from main_classwise import *

def run_all_classes():
    print("Running All Classes Mode...")
    train_all_classes()

def run_zeroshot():
    print("Running Zero-Shot Mode...")
    train_zero_shot()


if __name__ == "__main__":
    choice = input("Choose mode:\n1 - All Classes\n2 - Zero-Shot\nEnter your choice: ")

    if choice == "1":
        run_all_classes()
    elif choice == "2":
        run_zeroshot()
    else:
        print("Invalid choice. Please restart and enter 1 or 2.")
