import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(conf_mat, 
                          class_names,
                          interactive = False,
                          path = "",
                          title=""):
    """
      Plot the confusion matrix
    
      Args:
        conf_mat: confusion matrix
        class_names: class names
        interactive: flag to show interactive the plot
        path: where to save the plot. It will not be saved, if kept empty 
    """
    
    sns.set()
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    #annot=True to annotate cells
    sns.heatmap(conf_mat, annot=True, ax = ax, cmap="YlGnBu", fmt='g',
                xticklabels= class_names, yticklabels= class_names)
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix '+title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

#    # try to put the class names on the axes
#    # this seems to work when there "not so many" classes only...
#    try :
#        ax.xaxis.set_ticklabels(class_names)
#        ax.yaxis.set_ticklabels(class_names)
#    except:
#        print(cls.bcolors.WARNING
#              + "plot_functions::plot_confusion_matrix:: Warning: error initialising the plot axes"
#              + cls.bcolors.ENDC)
        
    # show the plot, if required
    if interactive :
        plt.show()

    # save the plot, if required
    if path != "" :

        # build the complete path where to save the plot
        path_full = os.path.join(path, title+"confusion_matrix.png")
        
        # save the plot
        plt.savefig(path_full)

    # close the current plot
    plt.close()

    return

matplotlib.use('tkagg')
