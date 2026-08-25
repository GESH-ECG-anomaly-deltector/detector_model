from load_dataset import *



if not "dataset" in os.listdir():
    ldob=load_dataset()
    ldob.load_dataset()
    ldob.denoise_signals()
    ldob.reduce_sample_rate()
    ldob.define_labels()
    ldob.define_8_superclasses()
    ldob.one_hot_encodding()

    signals, labels=ldob.get_signals_and_labels()

    if "dataset" in os.listdir():
        shutil.rmtree("dataset")
    os.mkdir("dataset")
    np.save("dataset/signals.npy", signals)
    np.save("dataset/labels.npy",  labels)

else:
    signals=np.load("dataset/signals.npy")
    labels=np.load("dataset/labels.npy")


print("signals.shape:  ", signals.shape)
print("labels.shape:   ", labels.shape)