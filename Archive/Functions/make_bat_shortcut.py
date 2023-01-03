import sys, os
# import win32com.client
# import pythoncom

class Error(Exception): pass

def _find(pathname, matchFunc=os.path.isfile):
    for dirname in sys.path:
        candidate = os.path.join(dirname, pathname)
        if matchFunc(candidate):
            return candidate
    raise Error("Can't find file %s" % pathname)

def change_capital(string_:str)->str:
    temp_list = list(string_)
    temp_list[0] = temp_list[0].upper()
    string_ = ''.join(temp_list)
    return string_

def findFile(pathname):
    temp = _find(pathname)
    return change_capital(temp)

def findDir(path):
    temp = _find(path, matchFunc=os.path.isdir)
    return change_capital(temp)