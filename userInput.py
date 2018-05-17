from pandas import read_excel
import scipy
from scipy.stats import ttest_ind
import test
import numpy
import copy
"""
python userInput.py san_jose.xlsx
O
1950,1970,2006
white
"""

class user_input:
    def output(self, data_matrix, columns_with_data):
        row_names_and_numbers = {}
        for y in range(5, len(data_matrix)):
            temp_string = ""
            for column in columns_with_data:
                s = str(data_matrix[y][columns_with_data[column]])
                if (s != 'nan'):
                    temp_string = temp_string + s.replace(" ", "") + " "
            row_names_and_numbers[temp_string.lower()] = y

        print("Please input values without any spaces")

        years = input("Input years separated by commas (or use 'all' without quotation marks for all years): ")
        year_parse = years.split(",")

        data_sets = input("Input data sets separated by commas: ")
        data_sets.replace(" ", "")
        data_parse = data_sets.split(",")

        if years == 'all':
            year_parse = []
            for key in columns_with_data:
                year_parse.append(key)

        for name in row_names_and_numbers:
            for c in columns_with_data:
                for year in year_parse:
                    if year in c:
                        name_row = row_names_and_numbers[name]
                        column_row = columns_with_data[c]
                        for d in data_parse:
                            if d.lower() in name:
                                temp_string = str(data_matrix[name_row][column_row]).lstrip(' ')
                                temp_string = ' '.join(temp_string.split())
                                if temp_string == 'nan':
                                    print(c + " is missing data at row " + str(name_row + 1) + " in the excel file")
                                if temp_string.isupper() and temp_string != 'nan':
                                    data_string = str(data_matrix[name_row][column_row + 1])
                                    print(c + ": " + temp_string + ": " + data_string)
                                if not temp_string.isupper() and temp_string != 'nan':
                                    y = name_row
                                    temp_string_internal = str(data_matrix[y][column_row])
                                    data_string = str(data_matrix[name_row][column_row + 1])
                                    while y >= 0 and not temp_string_internal.isupper():
                                        y = y - 1
                                        temp_string_internal = str(data_matrix[y][column_row])
                                    print(c + ": " + temp_string_internal + ": " + temp_string + ": " + data_string)

    def correlation(self):
        list_of_keys = list(self.dict_of_data)
        for x in range(0, len(list_of_keys)):
            print(str(x+1) + ": " + str(list_of_keys[x]))
        first_pick = input("Pick the 'x' data set (use the numbers): ")
        first_pick = int(first_pick) -1
        print("You picked " + list_of_keys[first_pick])
        second_pick = input("Pick the 'y' data set (use the numbers): ")
        second_pick = int(second_pick) -1
        print("You picked "+ list_of_keys[second_pick])
        self.nan_checker(list_of_keys[first_pick], self.dict_of_data[list_of_keys[first_pick]])
        self.nan_checker(list_of_keys[second_pick], self.dict_of_data[list_of_keys[second_pick]])
        test.correlation(list_of_keys[first_pick],list_of_keys[second_pick] ,self.dict_of_data[list_of_keys[first_pick]], self.dict_of_data[list_of_keys[second_pick]])

    def nan_checker(self,name,list):
        """
        :param name: String for name of list
        :param list: the list of values
        :return: none
        """
        nan_counter = 0
        for x in list:
            if str(x) == 'nan':
                nan_counter += 1
        if nan_counter >= 3:
            print(name + " is not a reliable data set! It is missing " + str(nan_counter) + " pieces of data!")

    def cut_down_predictors(self, x_frame):
        max_value = 500
        max_allowed_value = 5
        while(max_value > max_allowed_value):
            vif_calc = test.vif_cal(x_frame)
            max_value_key = max(vif_calc.keys(), key=lambda p: vif_calc[p])
            max_value = vif_calc[max_value_key]
            if (max_value > max_allowed_value):
                x_frame = x_frame.drop(max_value_key, axis=1)
            test.vif_cal(x_frame)
        return x_frame

    # def smart_nan_filler(self, array):
    #     for x in range(0, len(array)):
    #         if str(array[x]) == 'nan':
    #             one, two = self.get_closest_neighbors(x, array)
    #             max_pos = max([one, two], key=lambda p: array[p])
    #             min_pos = min([one, two], key=lambda p:array[p])
    #             slope = (array[max_pos] - array[min_pos]) / (max_pos-min_pos)
    #             addenum = (x - one) * slope
    #             array[x] = round(array[one] + addenum)
    #     return array
    #
    # def get_closest_neighbors(self, nan_index, array):
    #     minus = nan_index - 1
    #     plus = nan_index + 1
    #     return_value = 500
    #     return_value_2 = 500
    #     while return_value == 500:
    #         if minus >= 0 and str(array[minus]) != 'nan':
    #             return_value = minus
    #             break
    #         if plus < len(array) and str(array[plus]) != 'nan':
    #             return_value = plus
    #             break
    #         if minus < 0 and plus > len(array):
    #             break
    #         minus = minus - 1
    #         plus = plus + 1
    #
    #     while return_value_2 == 500:
    #         if minus >= 0 and str(array[minus]) != 'nan' and minus!=return_value:
    #             return_value_2 = minus
    #             break
    #         if plus < len(array) and str(array[plus]) != 'nan' and plus!= return_value:
    #             return_value_2 = plus
    #             break
    #         if minus < 0 and plus > len(array):
    #             break
    #         minus = minus - 1
    #         plus = plus + 1
    #
    #     return return_value, return_value_2

    def smart_nan_filler(self, array):
        list_of_positions = []
        for x in range(0, len(array)):
            list_of_positions.append(x)

        max_position = array.index(float(numpy.nanmax(array)))
        min_position = array.index(float(numpy.nanmin(array)))

        slope = (array[max_position] - array[min_position]) / (max_position - min_position)

        for x in range(0, len(array)):
            if str(array[x]) == 'nan':
                neighbor_position = self.get_closest_neighbor(x, array)
                addenum = (x - neighbor_position) * slope
                array[x] = round(array[neighbor_position] + addenum)
        return array

    def get_closest_neighbor(self, nan_index, array):
        minus = nan_index - 1
        plus = nan_index + 1
        return_value = 500
        while return_value == 500:
            if minus >= 0 and str(array[minus]) != 'nan':
                return_value = minus
                break
            if plus < len(array) and str(array[plus]) != 'nan':
                return_value = plus
                break
            if minus < 0 and plus > len(array):
                break
            minus = minus - 1
            plus = plus + 1
        return return_value

    def predict(self):
        list_of_keys = list(self.dict_of_data)
        list_of_keys.remove('Years')
        for x in range(0, len(list_of_keys)):
            print(str(x + 1) + ": " + str(list_of_keys[x]))
        first_pick = input("Pick the 'y' data set (use the numbers): ")
        first_pick = int(first_pick) - 1
        print("You picked " + list_of_keys[first_pick])

        x = list_of_keys[first_pick]

        array_x = self.smart_nan_filler(self.dict_of_data[x])

        self.nan_checker(x, self.dict_of_data[x])
        x = test.list_of_lists_to_dataframe([array_x], [x])
        #x.index = self.dict_of_data['Years']


        # Future grab difference in previous and add it to previous
        # x3 is Nan so x2-x1 = y
        # x3 = x2 + y

        # The Y data is just the years
        y = 'Years'
        #self.nan_checker(y, self.dict_of_data[y])
        y = test.list_of_lists_to_dataframe([self.dict_of_data[y]], [y])
        #y = y.fillna(y.mean())
        y.index = self.dict_of_data['Years']

        year = input("Which year would you like to predict?: ")
        year_int = float(year)

        a = -500
        b = -500
        if year_int > 2010 or year_int < 1950:
            a, b= self.better_prediction(x, year_int)

        test.predict(x,y, year_int,a,b)


    def linear_regression(self):
        list_of_keys = list(self.dict_of_data)
        list_of_keys.remove('Years')

        for x in range(0, len(list_of_keys)):
            key_string = list_of_keys[x]
            self.dict_of_data[key_string] = self.smart_nan_filler(self.dict_of_data[key_string])
            print(str(x + 1) + ": " + str(list_of_keys[x]))

        second_pick = input("Pick the dependent variable (use the numbers): ")
        second_pick = int(second_pick) - 1
        y_name = list_of_keys[second_pick]
        print("You picked " + y_name)
        self.nan_checker(y_name, self.dict_of_data[y_name])

        new_data = []
        data_names = []

        for x in range (0, len(list_of_keys)):
            if x != second_pick:
                new_data.append(self.dict_of_data[list_of_keys[x]])
                data_names.append(list_of_keys[x])

        new_data.append(self.dict_of_data[str(y_name)])
        data_names.append(y_name)

        new_data = test.list_of_lists_to_dataframe(new_data,data_names)

        x_frame = new_data.drop(new_data.columns[len(list_of_keys) - 1], axis=1)

        y_frame = new_data.drop(new_data.columns[0:len(list_of_keys) - 1], axis=1)

        change_data = input("Would you like to change your data? [S]tandardize, [N]ormalize, or [D]on't change?: ")
        if change_data.lower() == 's':
            standard = input("Standardize predictors(x), dependent variables(y), or both(b)?: ")
            if standard.lower() == 'x':
                x_frame = test.standardize_data(x_frame)
            elif standard.lower() == 'y':
                y_frame = test.standardize_data(y_frame)
            else:
                new_data = test.standardize_data(new_data)
                x_frame = new_data.drop(new_data.columns[len(list_of_keys) - 1], axis=1)
                y_frame = new_data.drop(new_data.columns[0:len(list_of_keys) - 1], axis=1)
        elif change_data.lower() == 'n':
            normalize = input("Normalize predictors(x), dependent variables(y), or both(b)?: ")
            if normalize.lower() == 'x':
                x_frame = test.normalize_data(x_frame)
            elif normalize.lower() == 'y':
                y_frame = test.normalize_data(y_frame)
            else:
                new_data = test.normalize_data(new_data)
                x_frame = new_data.drop(new_data.columns[len(list_of_keys) - 1], axis=1)
                y_frame = new_data.drop(new_data.columns[0:len(list_of_keys) - 1], axis=1)

        x_frame = self.cut_down_predictors(x_frame)
        print(x_frame)
        result = test.stepwise_selection(x_frame, y_frame)

        if not result:
            print("No correlation found!")
        else:
            test.regression_table(x_frame, result, y_frame)
        self.dict_of_data = self.old_dict
        self.old_dict = copy.deepcopy(self.dict_of_data)



    def manual_regression(self, mode):
        list_of_keys = list(self.dict_of_data)
        list_of_keys.remove('Years')

        for x in range(0, len(list_of_keys)):
            key_string = list_of_keys[x]
            self.dict_of_data[key_string] = self.smart_nan_filler(self.dict_of_data[key_string])
            print(str(x + 1) + ": " + str(list_of_keys[x]))

        new_data = []
        names = []

        x = input("Input Predictor: ")
        x = int(x) - 1
        new_data.append(self.dict_of_data[list_of_keys[x]])
        names.append(list_of_keys[x])

        get_predictors = True
        while get_predictors:
            x = input("More predictors? (Input [N]o or a number): ")
            if x.lower() == 'n':
                get_predictors = False
                break
            else:
                x = int(x) - 1
                new_data.append(self.dict_of_data[list_of_keys[x]])
                names.append(list_of_keys[x])


        y = input("Input Dependent Variable: ")
        y = int(y) -1
        new_data.append(self.dict_of_data[list_of_keys[y]])
        names.append(list_of_keys[y])

        new_data = test.list_of_lists_to_dataframe(new_data, names)

        x_frame = new_data.drop(new_data.columns[len(names)-1], axis=1)
        y_frame = new_data.drop(new_data.columns[0:len(names)-1], axis=1)


        change_data = input("Would you like to change your data? [S]tandardize, [N]ormalize, or [D]on't change?: ")
        if change_data.lower() == 's':
            standard = input("Standardize predictors(x), dependent variables(y), or both(b)?: ")
            if standard.lower() == 'x':
                x_frame = test.standardize_data(x_frame)
            elif standard.lower() == 'y':
                y_frame = test.standardize_data(y_frame)
            else:
                new_data = test.standardize_data(new_data)
                x_frame = new_data.drop(new_data.columns[len(names)-1], axis=1)
                y_frame = new_data.drop(new_data.columns[0:len(names)-1], axis=1)
        elif change_data.lower() == 'n':
            normalize = input("Normalize predictors(x), dependent variables(y), or both(b)?: ")
            if normalize.lower() == 'x':
                x_frame = test.normalize_data(x_frame)
            elif normalize.lower() == 'y':
                y_frame = test.normalize_data(y_frame)
            else:
                new_data = test.normalize_data(new_data)
                x_frame = new_data.drop(new_data.columns[len(names)-1], axis=1)
                y_frame = new_data.drop(new_data.columns[0:len(names)-1], axis=1)

        if mode == 'm' :
            names.remove(names[len(names)-1])
            test.regression_table(x_frame, names, y_frame)
        elif mode == 't':
            test.train_and_test(x_frame, y_frame)
        self.dict_of_data = self.old_dict
        self.old_dict = copy.deepcopy(self.dict_of_data)

    def __init__(self, file_name, dict_of_data):
        """
        :param file_name: String representing the excel file's name
        :param dict_of_data: the dictionary of data
        """
        self.dict_of_data = dict_of_data
        self.old_dict = copy.deepcopy(dict_of_data)
        data = read_excel(file_name, header=None)
        data_matrix = data.as_matrix()

        # key = name of column  #, value = column number
        columns_with_data = {}

        for x in range(0, len(data_matrix[0])):
            s = str(data_matrix[4][x])
            if s != 'nan':
                columns_with_data[s] = x

        exit = False
        while not exit:
            command = input("Enter your wish: [O]utput, [C]orrelation, [L]inear Regression, [M]anual Regression, [T]rain and Test, [P]redict, or [Q]uit: ")
            if command.lower() == 'o':
                self.output(data_matrix, columns_with_data)
            if command.lower() == 'c':
                self.correlation()
            if command.lower() == 'l':
                self.linear_regression()
            if command.lower() == 'p':
                self.predict()
            if command.lower() == 'm':
                self.manual_regression('m')
            if command.lower() == 't':
                self.manual_regression('t')
            #below is only used for testing, not sure if user should have access to it or not
            if command.lower() == 'tt':
                print("Average best number of training data: " + str(self.test_for_absolute_best_training_number()))
            if command.lower() == 'q':
                exit = True

    def better_prediction(self, y_frame, year):
        tested_predictors = []

        pred_1 = -500
        pred_2 = -500

        y_name = list(y_frame)[0]
        list_of_keys = list(self.dict_of_data)
        list_of_keys.remove(y_name)
        list_of_keys.remove('Years')

        year_frame = self.dict_of_data['Years']
        year_frame = test.list_of_lists_to_dataframe([year_frame], [y_name])

        data = []
        names = []

        for x in list_of_keys:
            self.dict_of_data[x] = self.smart_nan_filler(self.dict_of_data[x])
            data.append(self.dict_of_data[x])
            names.append(x)

        x_frame = test.list_of_lists_to_dataframe(data, names)


        x_frame = self.cut_down_predictors(x_frame)
        included = test.stepwise_selection(x_frame,y_frame)

        if included:
            for x in included:
                temp_frame = test.pd.DataFrame(x_frame[x])
                regr = test.LinearRegression()
                num = test.get_best_training_number(year_frame, temp_frame)
                temp_train = temp_frame.drop(temp_frame.index[num: len(temp_frame.index)])
                year_train = year_frame.drop(year_frame.index[num: len(year_frame.index)])

                regr.fit(year_train,temp_train)
                prediction = regr.predict(year)
                tested_predictors.append(prediction[0][0])

            approved_data = test.pd.DataFrame(x_frame[included])
            regr = test.LinearRegression()
            regr.fit(approved_data, y_frame)
            prediction = regr.predict([tested_predictors])
            pred_1 = prediction[0][0]

            regr = test.LinearRegression()
            best_num = test.get_best_training_number(approved_data, y_frame)
            approved_train = approved_data.drop(approved_data.index[best_num: len(approved_data.index)])
            y_train = y_frame.drop(y_frame.index[best_num: len(y_frame.index)])
            regr.fit(approved_train, y_train)
            prediction = regr.predict([tested_predictors])
            pred_2 = prediction[0][0]

        self.dict_of_data = self.old_dict
        self.old_dict = copy.deepcopy(self.dict_of_data)
        return pred_1, pred_2

    def get_number_of_number_in_list(self, array, num):
        return_number = array.count(num)
        print("Amount of ", end= "")
        print(num, end= "")
        print(" is " + str(return_number))
        return return_number

    def test_for_absolute_best_training_number(self):
        list_of_keys = list(self.dict_of_data)
        list_of_keys.remove('Years')
        for x in range(0, len(list_of_keys)):
            key_string = list_of_keys[x]
            self.dict_of_data[key_string] = self.smart_nan_filler(self.dict_of_data[key_string])
        list_of_numbers = []
        for x in list_of_keys:
            for y in list_of_keys:
                if x != y:
                    x_frame = self.dict_of_data[x]
                    x_frame = test.list_of_lists_to_dataframe([x_frame], [x])
                    y_frame = self.dict_of_data[y]
                    y_frame = test.list_of_lists_to_dataframe([y_frame], [y])
                    list_of_numbers.append(test.get_best_training_number(x_frame, y_frame))
        list_of_best = set(list_of_numbers)
        self.dict_of_data = self.old_dict
        self.old_dict = copy.deepcopy(self.dict_of_data)
        best_of_best = max(list_of_best, key=lambda p: self.get_number_of_number_in_list(list_of_numbers, p))
        return best_of_best