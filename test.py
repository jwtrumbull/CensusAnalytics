import argparse
from pandas import read_excel
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import userInput
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import re

def get_arguments():
    '''
    Parse and validate the command line arguments
    :return: The arguments in order
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('file_name',
                        help = 'the name of the file to open')

    arguments = parser.parse_args()

    file_name = arguments.file_name
   # census = arguments.census
    # start_row = arguments.start_row
    # end_row = arguments.end_row
    # start_column = arguments.start_column
    # end_column = arguments.end_column
    return file_name

def get_year(input_year):
    """
    :param input_year: the year the user wants to get
    :return:
    """
    if input_year == '1950':
        return '1950Census'
    if input_year == '1960':
        return '1960Census'
    if input_year == '1970':
        return '1980Census'
    if input_year == '1980':
        return '1980Census'
    if input_year == '1990':
        return '1960Census'
    if input_year == '2000':
        return 'Census2000'
    if input_year == '2010':
        return 'Census2000'
    if input_year == '2006-2010':
        return '2006-2010AmericanCommunitySurvey'

def main():
    file_name = get_arguments()
    data = read_excel(file_name, header =  None)
    data_matrix = data.as_matrix()
    #print(data_matrix)
    headers = {}
    #headers = {Census name: {totalpop: '100', 'RACE': {} }}
    for x in range(0, 36):
        subHeads = {}
        s = str(data_matrix[4][x])
        if s != 'nan':
            for y in range(5,143):
                subhead = str(data_matrix[y][x])
                if subhead != 'nan':
                    subheadx = str(data_matrix[y][x+1])
                    if subheadx == 'nan':
                        subsub = {}
                        y = y+1
                        while y <143 and not str(data_matrix[y][x]).isupper():
                            if str(data_matrix[y][x]) != 'nan':
                                subsub[str(data_matrix[y][x]).replace(" ", "")] = str(data_matrix[y][x+1]).replace(" ", "")
                            y = y + 1
                        subHeads[subhead.replace(" ", "")] = subsub
                    if subheadx != 'nan':
                        subHeads[subhead.replace(" ", "")] = subheadx.replace(" ", "")
        headers[s.replace(" ", "")] = subHeads
    #print(headers['1960Census']['TOTALPOPULATION'])

    parameters = ['TOTALPOPULATION']
    data = {'Years': [], 'Total Population': [], 'White': [], 'Black': [], 'Asian': [], 'Indian': [], 'Hispanic': [],
            'Male': [], 'Female': [], 'Under 5 years': [], '5 to 17 years': [],
            '18 to 64 years': [], '65 years and over': [], 'Median age': [], 'Persons per household': [],
            'Total households': [], 'Employed': [], 'Median household income': []}
    #print(headers.keys())
    inputyear = '2010'
    inputparam = 'Male'
    check1 = inputyear + 'Census'
    check2 = 'Census' + inputyear
    count = 0
    for key, value in headers.items():
        if key == '2006-2010AmericanCommunitySurvey':
            break
        if isinstance(value, dict): # if it is a dictionary
            if 'nan' not in key:
                year = re.sub('[^0-9]','',key)
                data['Years'].append(float(year))
            for k, v in value.items():
                if k == 'TOTALPOPULATION': # change to input later
                    data['Total Population'].append(float(v))
                if k == 'Male':
                    data['Male'].append(float(v))
                if k == 'White':
                    data['White'].append(float(v))
                if k == 'Nativewhite':
                    nativewhite = float(v)
                if k == 'Foreign-bornwhite':
                    numofwhite = nativewhite + float(v)
                    data['White'].append(numofwhite)
                if k == 'WHITEPERSONSOFSPANISH\nSURNAME' or k == 'PERSONSOFSPANISH\nORIGINORDESCENT' or k == 'SPANISHORIGIN' or k == 'TOTALHISPANIC' or k == 'HispanicorLatino(ofanyrace)':
                    data['Hispanic'].append(float(v))
                if k == 'Asian':
                    data['Asian'].append(float(v))
                if k == 'Indian' or k == 'Indians' or k == 'AmericanIndian,Eskimo,andAleut' or k == 'AmericanIndian,Eskimo,or\nAleut' or k == 'AmericanIndianandAlaska\nNative':
                    data['Indian'].append(float(v))
                if k == 'Female':
                    data['Female'].append(float(v))
                if k == 'Under5years':
                    data['Under 5 years'].append(float(v))
                if k == '5to17years' or k == '5to19years': # probably don't used this one
                    data['5 to 17 years'].append(float(v))
                if k == '18to64years' or k == '20to64years': # or this one
                    data['18 to 64 years'].append(float(v))
                if k == '65yearsandover':
                    data['65 years and over'].append(float(v))
                if k == 'Medianage':
                    data['Median age'].append(float(v))
                if k == 'Personsperhousehold' or k == 'Populationperhousehold' or k == 'Averagehouseholdsize':
                    data['Persons per household'].append(float(v))
                if k == 'Totalhouseholds' or k == 'Households' or k == 'Totalhousehold' or k == 'Headofhousehold':
                    data['Total households'].append(float(v))
                if k == 'Employed' or k == 'Employedcivilianpopulation\n16yearsandover':
                    data['Employed'].append(float(v))
                if k == 'Familiesmedianincomedollars' or k == 'Incomeoffamiliesandunrelated\nindividuals' or k == 'Medianhouseholdincome' or k == 'Medianhouseholdincome\n(dollars)':
                    data['Median household income'].append(float(v))
            for i in data: # Inserts NaN when there is no data entered
                if len(data[i]) is not count:
                    data[i].append(np.nan)
            count = count + 1
    # print('p')
    # print(data)



    dataframe = pd.DataFrame(data, columns=data.keys())

    user = userInput.user_input(file_name, data)  # the userInput class

    #print(dataframe)
    numofyears = 7
    if inputyear == '1950':
        inputyearfind = 0
    if inputyear == '1960':
        inputyearfind = 1
    if inputyear == '1970':
        inputyearfind = 2
    if inputyear == '1980':
        inputyearfind = 3
    if inputyear == '1990':
        inputyearfind = 4
    if inputyear == '2000':
        inputyearfind = 5
    if inputyear == '2010':
        inputyearfind = 6

   # print('Number of ' + inputparam + ' in ' + inputyear + ' is', dataframe[inputparam][inputyearfind])

    xmin = dataframe['Male'].min()

    xmax = dataframe['Male'].max()
    X = np.linspace(xmin, xmax, 100)


def correlation(name_x, name_y, data_x, data_y):
    """
    :param name_x: String for name of x
    :param name_y: String for name of y
    :param data_x: list
    :param data_y: list
    :return: nothing
    """
    plt.figure()
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.scatter(data_x, data_y)
    plt.show()

def regression_table(data_x, data_y):
    """
    :param data_x: A dataframe
    :param data_y: A dataframe
    :return: does not return anything
    """

    """
    #lm = smf.ols("Female ~ Male", data=dataframe).fit()
    xmin = min(data_x)
    xmax = max(data_x)
    X = np.linspace(xmin, xmax, 100)
    X = sm.add_constant(X)
    # params[0] is the intercept (beta0)
    # params[1] is the slope (beta1)
    #Y = lm.params[0] + lm.params[1] * X
    sns.regplot(data_x, data_y)
    plt.plot(X, Y, color="darkgreen")
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.plot(data_x, data_y)
    plt.show()
    """

    data_x = sm.add_constant(data_x)
    model = sm.OLS(data_y,data_x).fit()
    predictions = model.predict(data_x)
    print(model.summary())

def list_of_lists_to_dataframe(list,names):
    """
    :param list: a list of lists of variables; ex: [[1,2,3], [2,3,4]]
    :param names: a list of names for the individual lists; ex: ['a','b']
    :return: a proper dataframe with the names as the columns
    """
    new_list = []
    for y in range(0, len(list[0])):
        temp_list = []
        for x in list:
        #x is the individual arrays
            temp_list.append(x[y])
        new_list.append(temp_list)
    new_list = pd.DataFrame(new_list, columns=names)
    return new_list

def standardize_data(dataframe):
    """
    :param dataframe: DataFrame
    :return: dataframe of standardized data
    """
    scaler = StandardScaler()
    scaler.fit(dataframe)
    data= scaler.transform(dataframe)
    data = pd.DataFrame(data, columns=list(dataframe))
    return data

def normalize_data(dataframe):
    """
    :param dataframe: DataFrame
    :return: DataFrane of normalized data
    """
    normal = Normalizer()
    normal.fit(dataframe)
    data = normal.transform(dataframe)
    data = pd.DataFrame(data, columns=list(dataframe))
    return data

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out = 0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

if __name__ == '__main__':
    main()