import subprocess,sys,os

# def run_evt(args=None):
#     path=os.path.join(os.path.dirname(__file__),'lasp_train_evt')
#     subprocess.run(['bash',path])

def run_test_sequence(args=None):
    path=os.path.join(os.path.dirname(__file__),'test_squence.py')
    subprocess.run([sys.executable,path])

def run_otr(args=None):
    path=os.path.join(os.path.dirname(__file__),'impedence_calc.py')
    subprocess.run([sys.executable,path])

def run_ecsa(args=None):
    path=os.path.join(os.path.dirname(__file__),'ecsa_normal.py')
    subprocess.run([sys.executable,path])

def run_ecsa_dry(args=None):
    path=os.path.join(os.path.dirname(__file__),'ecsa_dry.py')
    subprocess.run([sys.executable,path])

def run_lsv(args=None):
    path=os.path.join(os.path.dirname(__file__),'lsv.py')
    subprocess.run([sys.executable,path])

def run_conclude(args=None):
    path=os.path.join(os.path.dirname(__file__),'conclude.py')
    subprocess.run([sys.executable,path])


def run_all(args=None):
    dirs=[dir for dir in os.listdir() if os.path.isdir(dir)]
    run_test_sequence()
    if "OTR" in dirs:
        run_otr()
    run_ecsa()
    run_ecsa_dry()
    run_lsv()
    run_conclude()