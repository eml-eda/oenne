# Optimized Execution of Neural Networks at the Edge 

PhD Course - Computer Engineering - Politecnico di Torino

**Authors**: Daniele Jahier Pagliari, Alessio Burrello

## Setup Instructions for Session #1

The ideal environment for running the Python notebooks (for the first Hands-on Session) is a local instance of Jupyter Lab. However, you will need a GPU to run all steps in a reasonable time. If you do not have access to one, please follow the instructions to use Kaggle or Google Colab instead.

### Instructions for Local Jupyter Lab

Clone the github repository from the command line:

```
git clone git@github.com:eml-eda/oenne.git
```

Move into the cloned folder:

```
cd oenne
```

Then, create a virtual environment to install packages locally:

```
python -m venv ./oenne_venv
```

And activate it:

```
source oenne_venv/bin/activate
```

Alternatively, you can use Conda as well if you prefer. Lastly, install the required packages with pip:

```
pip install -r requirements.txt
```

You are now ready to start the Jupyter Lab session. If you're running locally, this command should suffice:

```
jupyter lab
```

If you are in a remote machine, you may want to run this instead (where any port would work as long as it is not blocked by your firewall):

```
jupyter lab --ip='*' --port=58080 --no-browser
```

In the local scenario, a new browser window should popup automatically with the Jupyter Lab session. In the remote case, instead, you will see instructions in the terminal that suggest to access the server by opening a URL that looks like this:

```
http://localhost:58080/lab?token=<LONG_STRING>
```

Simply take that URL and paste it in your browser, replacing `localhost` with the IP or hostname of the remote server. You should now see the Jupyter Lab Interface. Simply open the first notebook `I_SuperNet.ipynb` and follow the instructions.



### Instructions for Kaggle

To run the notebooks in **Kaggle**, first create an account and associate it with your phone number (required to enable Internet access and GPU usage, max 30h per week). Then, create a new Notebook, and select: `File/Import Notebook`. Drag and drop the ipynb file that you want to run (starting from `I_SuperNet.ipynb` in the upload window.

Lastly in the right pane, under "Session Options", enable Internet by clicking on the toggle. Under "Accelerator", select "GPU P100".

### Instructions for Google Colab

To run the notebooks in **Google colab**, upload the corresponding `.ipynb` file to your Google Drive, then Right Click on it and select: `Open with/Google Colaboratory`.


### Common Notes for Kaggle and Colab 

If you're running on *either* Kaggle or Colab, the notebooks contain extra instructions and commands. Make sure to follow them.

Moreover, beware that each notebook will run in a **separate environment** from the others. Since each of the notebooks uses the outputs from the previous ones, you will have to make sure that these files are accessible "by hand" (uploading them in the respective cloud folders and setting all paths appropriately). See the instructions in the notebooks.

