from scipy.io import arff
import pandas as pd
import numpy as np
protocol_type_mapping = {b'tcp': 0,
                         b'udp': 1,
                         b'icmp': 2}
service_mapping = {b'aol': 0,
                   b'auth': 1, b'bgp': 2, b'courier': 3, b'csnet_ns': 4, b'ctf': 5, b'daytime': 6,
                   b'discard': 7, b'domain': 8, b'domain_u': 9, b'echo': 10, b'eco_i': 11, b'ecr_i': 12,
                   b'efs': 13, b'exec': 14, b'finger': 15, b'ftp': 16, b'ftp_data': 17, b'gopher': 18, b'harvest': 19,
                   b'hostnames': 20, b'http': 21, b'http_2784': 22, b'http_443': 23, b'http_8001': 24, b'imap4': 25,
                   b'IRC': 26, b'iso_tsap': 27, b'klogin': 28, b'kshell': 29, b'ldap': 30,
                   b'link': 30, b'login': 31, b'mtp': 32, b'name': 33,
                   b'netbios_dgm': 34, b'netbios_ns': 35, b'netbios_ssn': 36,
                   b'netstat': 37, b'nnsp': 38, b'nntp': 39, b'ntp_u': 40,
                   b'other': 41, b'pm_dump': 42, b'pop_2': 43, b'pop_3': 44, b'printer': 45, b'private': 46,
                   b'red_i': 47, b'remote_job': 48, b'rje': 49, b'shell': 50, b'smtp': 51, b'sql_net': 52,
                   b'ssh': 53, b'sunrpc': 54, b'supdup': 55, b'systat': 56,
                   b'telnet': 57, b'tftp_u': 58, b'tim_i': 59, b'time': 60,
                   b'urh_i': 61, b'urp_i': 62, b'uucp': 63, b'uucp_path': 64, b'vmnet': 65, b'whois': 66, b'X11': 67, b'Z39_50': 68}
flag_mapping = {b'OTH': 0, b'REJ': 1, b'RSTO': 2, b'RSTOS0': 3,
                b'RSTR': 4, b'S0': 5, b'S1': 6, b'S2': 7, b'S3': 8, b'SF': 9, b'SH': 10}
class_mapping = {b'normal':0, b'anomaly':1}
byte_to_int = {b'0': 0, b'1': 1}

def loadArffFromFile(path: str, to_predict: str = 'class'):
    """ 
    Parameters
    ----------
    path -> Path the the file
    to_predict -> Params that is our expected value

    Returns
    -------
    dataframe -> data frame object from pandas \n
    array_1 -> Numpy array of elements to predict \n
    array_2 -> Numpy array of prediction expected result
    """
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    return df

def split(dataframe, to_predict: str = 'class'):
    """ 
    Parameters
    ----------
    Dataframe, parameter which will be split

    Returns
    -------
    array of elements except a given class
    array of element to the cooresponding class
    """
    return np.array(dataframe.drop([to_predict], 1)), np.array(dataframe[to_predict])

def preprocessKDDDataSet(data):
    data.replace({'protocol_type': protocol_type_mapping}, inplace=True)
    data.replace({'service': service_mapping}, inplace=True)
    data.replace({'flag': flag_mapping}, inplace=True)
    data.replace({'land': byte_to_int}, inplace=True)
    data.replace({'logged_in': byte_to_int}, inplace=True)
    data.replace({'is_host_login': byte_to_int}, inplace=True)
    data.replace({'is_guest_login': byte_to_int}, inplace=True)
    data.replace({'class': class_mapping}, inplace=True)
    return data
