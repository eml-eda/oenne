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

In this second case, you will see instructions in the terminal that suggest to access the server by opening a URL that looks like this:

```
http://localhost:58080/lab?token=<LONG_STRING>
```

Simply take that URL and paste it in your browser, replacing `localhost` with the IP or hostname of the remote server. You should now see the Jupyter Lab Interface. Simply open the first notebook `I_SuperNet.ipynb` and follow the instructions.



### Instructions for Kaggle


### Instructions for Google Colab
