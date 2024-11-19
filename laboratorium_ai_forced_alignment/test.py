import main

INPUT_FILE = "../test_inputs/input_1.wav"
OUTPUT_FILE = "../test_outputs/output_1.json"

if __name__ == "__main__":
    main.load_model()
    main.process_audio(INPUT_FILE, OUTPUT_FILE)
