import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import argparse

def main(module_name):

    if module_name == "syn":
        # function to update the data
        def my_function(i):
            # get data
            train_loss_arr = np.load("synthesizer_loss/synthesizer_train_loss.npy")
            dev_loss_arr = np.load("synthesizer_loss/synthesizer_dev_loss.npy")

            # clear axis
            ax.cla()
            # plot cpu
            ax.plot(train_loss_arr)
            ax.plot(dev_loss_arr)
            ax.legend(["Train Loss", "Dev Loss"])
            ax.scatter(len(train_loss_arr) - 1, train_loss_arr[-1])
            ax.text(len(train_loss_arr)-1, train_loss_arr[-1], f"({len(train_loss_arr) - 1}, {train_loss_arr[-1]:.6})")
            ax.scatter(len(dev_loss_arr) - 1, dev_loss_arr[-1])
            ax.text(len(dev_loss_arr)-1, dev_loss_arr[-1], f"({len(dev_loss_arr) - 1}, {dev_loss_arr[-1]:.6})")
            # ax.set_ylim([0, 1])
            plt.xlabel("*100Steps")
            plt.ylabel("Loss")
            plt.title("Synthesizer Loss")
        # define and adjust figure
        fig, ax = plt.subplots()
        ax.set_facecolor('#DEDEDE')
        plt.xlabel("total steps")
        # animate
        ani = FuncAnimation(fig, my_function, interval=1000)
        plt.show()

    elif module_name == "voc":
        # function to update the data
        def my_function(i):
            # get data
            train_loss_arr = np.load("vocoder_loss/vocoder_train_loss.npy")
            dev_loss_arr = np.load("vocoder_loss/vocoder_dev_loss.npy")
            # clear axis
            ax.cla()
            # plot cpu
            ax.plot(train_loss_arr)
            ax.plot(dev_loss_arr)
            ax.legend(["Train Loss", "Dev Loss"])
            ax.scatter(len(train_loss_arr) - 1, train_loss_arr[-1])
            ax.text(len(train_loss_arr), train_loss_arr[-1]+0.1, f"({len(train_loss_arr) - 1}, {train_loss_arr[-1]:.6})")
            ax.scatter(len(dev_loss_arr) - 1, dev_loss_arr[-1])
            ax.text(len(dev_loss_arr), dev_loss_arr[-1]-0.1, f"({len(dev_loss_arr) - 1}, {dev_loss_arr[-1]:.6})")
            ax.set_ylim([0, 5])
            plt.xlabel("*100Steps")
            plt.ylabel("Loss")
            plt.title("Vocoder Loss")
        # define and adjust figure
        fig, ax = plt.subplots()
        ax.set_facecolor('#DEDEDE')
        plt.xlabel("total steps")
        # animate
        ani = FuncAnimation(fig, my_function, interval=1000)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", type=str, help= \
    "The model to show plot, model name is syn or voc")
    args = parser.parse_args()
    arg_dict = vars(args)
    try:
        main(arg_dict["model"])
    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")