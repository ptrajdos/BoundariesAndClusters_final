#if python-javabridge cannot find libjvm you may try to source this file
export CP_JAVA_HOME=$(readlink -f $JAVA_HOME )/