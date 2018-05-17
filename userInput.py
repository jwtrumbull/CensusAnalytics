
from pandas import read_excel
import test
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

    def linear_regression(self):
        list_of_keys = list(self.dict_of_data)
        for x in range(0, len(list_of_keys)):
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

        new_data = new_data.fillna(new_data.mean())

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

        result = test.stepwise_selection(x_frame, y_frame)

        if not result:
            print("No correlation found!")
        else:
            new_x = []
            new_x_names = []
            for r in result:
                new_x_names.append(r)
                new_x.append(self.dict_of_data[r])
            new_x = test.list_of_lists_to_dataframe(new_x, new_x_names)
            test.regression_table(new_x, y_frame)

    def __init__(self, file_name, dict_of_data):
        """
        :param file_name: String representing the excel file's name
        :param dict_of_data: the dictionary of data
        """
        self.dict_of_data = dict_of_data
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
            command = input("Enter your wish: [O]utput, [C]orrelation, [L]inear Regression, or [Q]uit: ")
            if command.lower() == 'o':
                self.output(data_matrix, columns_with_data)
            if command.lower() == 'c':
                self.correlation()
            if command.lower() == 'l':
                self.linear_regression()
            if command.lower() == 'q':
                exit = True