import os

PYTHONROOT = os.path.abspath(os.path.dirname(__file__))
SRCROOT = os.path.join(PYTHONROOT,"../","../","../")
PROJECTROOT = os.path.join(SRCROOT,"../")

LIBPATH = os.path.abspath( os.path.join(PROJECTROOT,"lib"))
DISTPATH = os.path.abspath( os.path.join(PROJECTROOT,"dist"))
PREPATH = os.path.abspath( os.path.join(PROJECTROOT,"preload_classpath"))
DATAPATH = os.path.abspath( os.path.join(PROJECTROOT,"data"))
CONFIGPATH = os.path.abspath( os.path.join(PROJECTROOT,"configs"))
RESULSTPATH = os.path.abspath( os.path.join(PROJECTROOT,"results"))

CLASSPATH=[os.path.join(PREPATH,o) for o in os.listdir(PREPATH) if o.endswith('.jar')]
CLASSPATH+=sorted([os.path.join(LIBPATH,o) for o in os.listdir(LIBPATH) if o.endswith('.jar')])
CLASSPATH+=[os.path.join(DISTPATH,o) for o in os.listdir(DISTPATH) if o.endswith('.jar')]
