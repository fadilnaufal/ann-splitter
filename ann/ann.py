# ann.py : Python code for Artificial Neural Network Modelling 
import time
start = time.time()
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tabulate import tabulate as tab
import colorama
colorama.init(wrap=True)

class TrainingStateLogger(Callback):
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data
        
    def on_train_begin(self, logs=None):
        self.epochs = []
        self.learning_rates = []
        self.gradients = []
        self.val_checks = []
        self.squared_params = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        
        # Learning rate
        lr = self.model.optimizer.learning_rate.numpy()
        self.learning_rates.append(lr)
        
        # Gradients
        X_train, y_train = self.train_data
        with tf.GradientTape() as tape:
            y_pred = self.model(X_train, training=True)
            loss = self.model.compute_loss(X_train, y_train, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        grad_norm = np.sum([tf.norm(grad).numpy() for grad in gradients if grad is not None])
        self.gradients.append(grad_norm)
        
        # Validation checks (we use validation loss here as a proxy)
        val_loss = logs.get('val_loss')
        self.val_checks.append(val_loss)
        
        # Sum of squared parameters
        squared_sum = np.sum([np.sum(np.square(w.numpy())) for w in self.model.trainable_weights])
        self.squared_params.append(squared_sum)

def printf(message, file):
    print(message) 
    print(message, file=file) 

# Open the file in write mode
with open('ann.out', 'w') as f:
    #printf("Hello, world!", f)

    # Function to show the ANN functions including ReLU activations
    def show_ann_functions_with_relu(model, inp_list, out_list, head_train, scaler_x, scaler_y):
        layer_weights = [layer.get_weights()[0] for layer in model.layers]
        layer_biases = [layer.get_weights()[1] for layer in model.layers]
        
        # Get the MinMaxScaler parameters for input and output
        x_min = scaler_x.data_min_
        x_range = scaler_x.data_range_
        y_min = scaler_y.data_min_
        y_range = scaler_y.data_range_
        
        # Normalize the input features
        normalized_inputs = [(f"((x({i + 1}) - {x_min[i]}) / {x_range[i]})") for i in range(len(x_min))]
        
        # Generate the expressions for each layer
        layer_expressions = []
        for i, (weights, biases) in enumerate(zip(layer_weights, layer_biases)):
            expressions = []
            if i == 0:
                # Input layer
                for j in range(weights.shape[1]):
                    expr = " + ".join([f"({weights[k, j]} * {normalized_inputs[k]})" for k in range(weights.shape[0])])
                    expr = f"max(0, {expr} + {biases[j]})"
                    expressions.append(expr)
            elif i < len(layer_weights) - 1:
                # Hidden layers
                for j in range(weights.shape[1]):
                    expr = " + ".join([f"({weights[k, j]} * {layer_expressions[i-1][k]})" for k in range(weights.shape[0])])
                    expr = f"max(0, {expr} + {biases[j]})"
                    expressions.append(expr)
            else:
                # Output layer
                for j in range(weights.shape[1]):
                    expr = " + ".join([f"({weights[k, j]} * {layer_expressions[i-1][k]})" for k in range(weights.shape[0])])
                    expr = f"{expr} + {biases[j]}"
                    expr = f"({expr}) * {y_range[j]} + {y_min[j]}"
                    expressions.append(expr)
            layer_expressions.append(expressions)
        
        # Show the functions
        functions = []
        for out_idx in range(len(out_list)):
            input_names = [head_train[inp_list[i] - 1] for i in range(len(inp_list))]
            function_str = f"fn{out_idx+1} = {layer_expressions[-1][out_idx]}"
            functions.append((function_str, input_names))
        
        printf("\nANN Output Equations with ReLU Activation Function", f)
        printf('-'*60,f)
        for func, input_names in functions:
            printf("\n" + func, f)
        
        printf('',f)
        for out_idx in range(len(out_list)):
            function_legend = f"fn{out_idx+1} = {head_train[out_list[out_idx] - 1]}"
            printf(function_legend, f)
        
        printf(f"\nwhere {', '.join([f'x({i + 1}) = {name}' for i, name in enumerate(input_names)])}", f)

    def name_ext(nameext):
        name = nameext.rsplit('.', 1)[0]
        ext = nameext.rsplit('.', 1)[1]
        return name, ext

    # Customized banner
    banner = "   ann.py   \n  \         \n --o--o-->  \n  /         "
    printf('\n\033[1m'+'\033[44m'+banner+'\033[0m\n',f)
    printf("Python Program for Artificial Neural Network Modelling",f)

    fname1 = input("Enter the csv or xlsx file for Data Training : ") #"data_training.xlsx"
    fname2 = input("Enter the csv or xlsx file for Data Testing  : ") #"data_testing.xlsx"

    error_msg = '\n\033[1m'+'\033[91m'+f"ERROR: The files is not suitable or available in the current directory."+'\033[0m\n-fn \n'
    error_msg_2 = '\n\033[1m'+'\033[91m'+"ERROR: The input(s) you are entering is not valid."+'\033[0m\n-fn \n'

    # Name and extension
    try:
        extension1 = name_ext(fname1)
        extension2 = name_ext(fname2)
    except:
        printf(error_msg,f)
        quit()
    if not ((extension1[1] == "csv" and extension2[1] == "csv") or (extension1[1] == "xlsx" and extension2[1] == "xlsx")):
        printf(error_msg,f)
        quit()

    # Load the data
    if extension1[1] == "csv" and extension2[1] == "csv":
        try: open(fname1); open(fname2)
        except: printf(error_msg,f); quit()
        data_training = pd.read_csv(fname1)
        data_testing = pd.read_csv(fname2)
    elif extension1[1] == "xlsx" and extension2[1] == "xlsx":
        try: open(fname1); open(fname2)
        except: printf(error_msg,f); quit()
        data_training = pd.read_excel(fname1)
        data_testing = pd.read_excel(fname2)

    printf(f"\nLoading {fname1} and {fname2}.",f)

    head_train = data_training.columns
    head_train = list(head_train)
    head_train_2 = []
    for i in head_train:
        simptxt = i[:9]+" .."
        head_train_2.append(simptxt)

    head_test = data_testing.columns
    head_test = list(head_test)

    # Exit if headers are not the same
    if head_test != head_train:
        printf(f"The columns from {fname1} are not the same as the columns from {fname2}",f)
        quit()

    data = {header: i + 1 for i, header in enumerate(head_train_2)}

    # Create a DataFrame with the data
    df = pd.DataFrame(list(data.items()), columns=["  Column Name  ", "Index"])

    # Show the DataFrame using tabulate with proper headers
    printf('',f)
    printf(tab(df, headers="keys", tablefmt="pretty", showindex=False),f)
    printf('',f)

    # Proceed with the rest of the code
    try:
        inp = input("Please select the INPUT column Index(s) (separated with comma)             : ") #"2,3,4" #
        out = input("Please select the OUTPUT column Index(s) (separated with comma)            : ") #"5,6,7" #
        no_of_hidden_layers = int(input("Please enter the number of hidden layers (1 ~ ∞)                           : ")) #1 #
        no_of_nodes_hidden = input("Please enter the number of uniform neurons per hidden layer (1 ~ ∞)        : ") #10 #
        if no_of_hidden_layers > 1 and no_of_nodes_hidden == "custom":
            no_of_nodes_hidden_2 = input("Please enter the number of neurons per hidden layer (separated with comma) : ") #10 #
            list_no_of_nodes_hidden = [int(num) for num in no_of_nodes_hidden_2.split(",")]
        elif no_of_hidden_layers == 1 and no_of_nodes_hidden == "custom": raise ValueError(''); quit()
        else: 
            no_of_nodes_hidden = int(no_of_nodes_hidden)
            list_no_of_nodes_hidden = [no_of_nodes_hidden for i in range(no_of_hidden_layers)]
        training_cycles = int(input("Please enter the number of training cycles (1 ~ ∞)                         : ")) #100 #
        if no_of_hidden_layers != len(list_no_of_nodes_hidden):
            raise ValueError('')
        printf('',f)
        printf(f"The selected INPUT column index(s)              : {inp}",f)
        printf(f"The selected OUTPUT column index(s)             : {out}",f)
        printf(f"The number of hidden layers                     : {no_of_hidden_layers}",f)
        printf(f"The number of uniform neurons per hidden layer  : {no_of_nodes_hidden}",f)
        if no_of_nodes_hidden == "custom":
            printf(f"The number of neurons per hidden layer          : {list_no_of_nodes_hidden}",f)
        printf(f"The number of training cycles                   : {training_cycles}",f)
    except:
        printf(error_msg_2,f)
        quit()
    printf('',f)

    # Strip and split the input
    inp = inp.strip().split(",")
    inp_list = [int(num) for num in inp]
    out = out.strip().split(",")
    out_list = [int(num) for num in out]

    X_train = data_training.iloc[:, [i - 1 for i in inp_list]].values
    Y_train = data_training.iloc[:, [i - 1 for i in out_list]].values
    X_test = data_testing.iloc[:, [i - 1 for i in inp_list]].values
    Y_test = data_testing.iloc[:, [i - 1 for i in out_list]].values

    # Normalization
    scaler_x = MinMaxScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    X_test_norm = X_test
    scaler_y = MinMaxScaler()
    Y_train = scaler_y.fit_transform(Y_train)
    Y_test = scaler_y.transform(Y_test)

    # Initialize and compile the model
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(units=list_no_of_nodes_hidden[0], activation='relu', input_dim=X_train.shape[1], name=f'hidden_layer_1'))

    # Adding additional hidden layers based on user input
    if no_of_hidden_layers > 1 and no_of_nodes_hidden != "custom":
        for iii in range(2, no_of_hidden_layers + 1):
            model.add(Dense(no_of_nodes_hidden, activation='relu', name=f'hidden_layer_{iii}'))
    elif no_of_hidden_layers > 1 and no_of_nodes_hidden == "custom":
        for iii in range(2, no_of_hidden_layers + 1):
            model.add(Dense(list_no_of_nodes_hidden[iii-1], activation='relu', name=f'hidden_layer_{iii}'))

    # Adding the output layer
    model.add(Dense(units=Y_train.shape[1], activation='linear', name='output_layer'))

    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training the ANN on the training data
    training_state_logger = TrainingStateLogger((X_train, Y_train))
    history = model.fit(X_train, Y_train, batch_size=10, epochs=training_cycles, validation_data=(X_test, Y_test), callbacks=[training_state_logger], verbose=1)
    printf('',f)

    # Show the weights of each layer
    printf('-' * 100,f)
    for layer in model.layers:
        weights, biases = layer.get_weights()
        printf(f'Layer: {layer.name}',f)
        printf(f'Weights:\n{weights}',f)
        printf(f'Biases:\n{biases}',f)
        printf('-' * 100,f)

    # Evaluate the model
    loss = model.evaluate(X_test, Y_test)

    printf(f'Test Loss: {loss}',f)

    printf('',f)
    model.summary()
    
    X_total = np.vstack((X_train, X_test))
    Y_total = np.vstack((Y_train, Y_test))

    # Predicting the Test set results
    Y_pred = model.predict(X_total)
    Y_test_pred = model.predict(X_test).flatten()
    Y_train_pred = model.predict(X_train).flatten()

    # Un-Normalization
    Y_pred = scaler_y.inverse_transform(Y_pred)
    Y_test = scaler_y.inverse_transform(Y_test)
    X_test = scaler_x.inverse_transform(X_test)
    X_total = scaler_x.inverse_transform(X_total)
    Y_total = scaler_y.inverse_transform(Y_total)

    printf('',f)
    # Calculate RMSE
    rmse = np.sqrt(np.mean((Y_pred - Y_total) ** 2, axis=0))
    printf(f'Root Mean Squared Error (RMSE) \t: {rmse}',f)

    # Calculate Relative Error
    relative_error = np.mean(np.abs((Y_pred - Y_total) / Y_total), axis=0)
    printf(f'Relative Error \t\t\t: {relative_error}',f)

    # Calculate Squared Error
    squared_error = np.mean((Y_pred - Y_total) ** 2, axis=0)
    printf(f'Squared Error \t\t\t: {squared_error}',f)

    printf('',f)
    # Calculate R^2
    printf("Coefficient of Determination (R^2)",f)
    r2_scores = []
    for i in range(len(out_list)):
        r2 = r2_score(Y_total[:, i], Y_pred[:, i])
        r2_scores.append(r2)
        printf(f'R^2 for output {i+1}\t: {r2}',f)

    # Overall R^2
    overall_r2 = np.mean(r2_scores)
    printf(f'Overall R^2\t\t: {overall_r2}',f)
    r2 = r2_score(Y_total, Y_pred)
    printf('',f)

    # Save the model
    model.save("ann_model.keras")

    show_ann_functions_with_relu(model, inp_list, out_list, head_train, scaler_x, scaler_y)
    printf("\nFor use in MATLAB, just copy the equation to the appropriate MATLAB function file.",f)
    printf("For use in spreadsheet, change the input variables x(1), x(2), x(3), ..., x(n) to the appropriate cell name.",f)
    print()

    # Create the table to display input features, actual output, and predicted output
    input_headers = [head_train[item - 1][:5] + '..' for item in inp_list]
    output_headers = []
    for item in out_list:
        output_headers.append(f"{head_train[item - 1][:5]} Act.")
        output_headers.append(f"{head_train[item - 1][:5]} Pred.")
    table_headers = input_headers + output_headers

    data_to_display = np.hstack((X_total, Y_total, Y_pred))
    display_rows = []
    for row in data_to_display:
        input_features = row[:len(inp_list)]
        actual_outputs = row[len(inp_list):len(inp_list) + len(out_list)]
        predicted_outputs = row[len(inp_list) + len(out_list):]
        combined_outputs = np.column_stack((actual_outputs, predicted_outputs)).flatten()
        display_rows.append(np.hstack((input_features, combined_outputs)))

    # Convert to DataFrame for tabulate and print tabulate
    df_display = pd.DataFrame(display_rows, columns=table_headers)
    df_display_rounded = df_display.round(4)
    printf("\nActual Testing Data and Predicted Output Table",f)
    printf(tab(df_display_rounded, headers='keys', tablefmt='pretty', showindex=False),f)

    # Export the DataFrame to a file based on input file extension
    output_file = "Act_vs_Pred_Table." + extension1[1]
    if extension1[1] == 'csv':
        df_display.to_csv(output_file, index=False)
    elif extension1[1] == 'xlsx':
        df_display.to_excel(output_file, index=False)
    printf(f"This table is exported to {output_file}",f)
    printf('',f)

    end = time.time()
    dur = end - start
    printf(f"Execution time : {dur} s",f)
    printf('\033[1m'+'\033[92m'+"Successfully creating Artificial Neural Network Model"+'\033[0m',f)
    printf("-fn\n",f)

# Sort the data by the feature used for x-axis to make the plot clearer
sorted_indices = np.argsort(X_total[:, 0])
X_total_sorted = X_total[sorted_indices]
Y_total_sorted = Y_total[sorted_indices]
Y_pred_sorted = Y_pred[sorted_indices]
x_axis_feature = head_train[inp_list[0] - 1]

# Visualize the ANN Architecture with varying opacity based on weights
fig, ax = plt.subplots(figsize=(8, 6))
layer_spacing = 1

# Input layer
input_layer_neurons = len(inp_list)
for i in range(input_layer_neurons):
    ax.scatter(0, i - (input_layer_neurons - 1) / 2, color='blue', label='Input' if i == 0 else '')

# Hidden layer
if no_of_hidden_layers > 0:
    hidden_layer_neurons = list_no_of_nodes_hidden
    total_layers = 1 + len(hidden_layer_neurons) + 1  # Input, hidden, output
    max_hidden_neurons = max(hidden_layer_neurons)

    for layer_idx, num_neurons in enumerate(hidden_layer_neurons, start=1):
        for neuron_idx in range(num_neurons):
            ax.scatter(layer_idx * layer_spacing, neuron_idx - (num_neurons - 1) / 2, color='green', label='Hidden' if layer_idx == 1 and neuron_idx == 0 else '')

    # Connect input layer to hidden layer
    weights_input_hidden = model.layers[0].get_weights()[0]
    for i in range(input_layer_neurons):
        for j in range(hidden_layer_neurons[0]):
            opacity = abs(weights_input_hidden[i, j]) / np.max(np.abs(weights_input_hidden))  # Adjust opacity based on weights
            ax.plot([0, 1], [i - (input_layer_neurons - 1) / 2, j - (hidden_layer_neurons[0] - 1) / 2], color='black', linewidth=0.5, alpha=opacity)

    # Connect hidden layers
    for layer_idx in range(len(hidden_layer_neurons) - 1):
        weights_hidden_hidden = model.layers[layer_idx + 1].get_weights()[0]
        for i in range(hidden_layer_neurons[layer_idx]):
            for j in range(hidden_layer_neurons[layer_idx + 1]):
                opacity = abs(weights_hidden_hidden[i, j]) / np.max(np.abs(weights_hidden_hidden))  # Adjust opacity based on weights
                ax.plot([layer_idx + 1, layer_idx + 2], [i - (hidden_layer_neurons[layer_idx] - 1) / 2, j - (hidden_layer_neurons[layer_idx + 1] - 1) / 2], color='black', linewidth=0.5, alpha=opacity)

# Output layer
output_layer_neurons = len(out_list)
for i in range(output_layer_neurons):
    ax.scatter(no_of_hidden_layers + 1, i - (output_layer_neurons - 1) / 2, color='red', label='Output' if i == 0 else '')

# Connect hidden layer to output layer
if no_of_hidden_layers > 0:
    weights_hidden_output = model.layers[-1].get_weights()[0]
    for i in range(hidden_layer_neurons[-1]):
        for j in range(output_layer_neurons):
            opacity = abs(weights_hidden_output[i, j]) / np.max(np.abs(weights_hidden_output))  # Adjust opacity based on weights
            ax.plot([no_of_hidden_layers, no_of_hidden_layers + 1], [i - (hidden_layer_neurons[-1] - 1) / 2, j - (output_layer_neurons - 1) / 2], color='black', linewidth=0.5, alpha=opacity)

# Set ANN Architecture plot
ax.set_xlabel('Layers')
ax.set_ylabel('Neurons')
ax.set_title('Artificial Neural Network Architecture')
ax.set_xlim(-0.5, no_of_hidden_layers + 1.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(layer_spacing))
ax.yaxis.set_major_locator(ticker.NullLocator())
ax.legend(loc='upper right')
fig.savefig('1_ANN_Architecture.png', dpi=500)
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.savefig('2_Model_Loss.png', dpi=500)
plt.show()

# Plot stacked error histogram
Y_test_er = Y_test.flatten()
Y_train_er = Y_train.flatten()
errors_test = Y_test_er - Y_test_pred
errors_train = Y_train_er - Y_train_pred
plt.hist([errors_train, errors_test], bins=30, label=['Training Errors', 'Testing Errors'], stacked=True, edgecolor='k')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
plt.xlabel('Error = Actual - Predicted')
plt.ylabel('Frequency')
plt.title('Error Histogram')
plt.legend()
plt.savefig('3_Error_Histogram.png', dpi=500)
plt.show()

# Plot training state metrics
epochs = training_state_logger.epochs
learning_rates = training_state_logger.learning_rates
gradients = training_state_logger.gradients
val_checks = training_state_logger.val_checks
squared_params = training_state_logger.squared_params

plt.figure(figsize=(12, 8))

# Plot gradients
ax1 = plt.subplot(4, 1, 1)
ax1.plot(epochs, gradients)
ax1.set_title('Gradient Norm (grad)')
ax1.set_ylabel('grad')
ax1.grid(True)
ax1.tick_params(labelbottom=False)

# Plot learning rates
ax2 = plt.subplot(4, 1, 2)
ax2.plot(epochs, learning_rates)
ax2.set_title('Learning Rate (mu)')
ax2.set_ylabel('mu')
ax2.grid(True)
ax2.tick_params(labelbottom=False)

# Plot validation checks (val_loss as proxy)
ax3 = plt.subplot(4, 1, 3)
ax3.plot(epochs, val_checks)
ax3.set_title('Validation Checks (val_f)')
ax3.set_ylabel('val_f')
ax3.grid(True)
ax3.tick_params(labelbottom=False)

# Plot sum of squared parameters
ax4 = plt.subplot(4, 1, 4)
ax4.plot(epochs, squared_params)
ax4.set_title('Sum of Squared Parameters (ssp)')
ax4.set_ylabel('ssp')
ax4.set_xlabel('Epoch')
ax4.grid(True)
ax4.tick_params(labelbottom=True)

plt.tight_layout(pad=4.0)  # Add padding between subplots
plt.savefig('4_Training_State.png', dpi=500)
plt.show()

see_vs_input = False
if see_vs_input == True:
    for indx in range(len(out_list)):
        # Plot the predicted and actual testing data using line plot
        plt.figure(figsize=(10, 6))

        # Use line plot with sorted data
        plt.plot(X_total_sorted[:, 0], Y_pred_sorted[:, indx], color='red', alpha=0.6, label='Predicted', linestyle='--')
        plt.plot(X_total_sorted[:, 0], Y_total_sorted[:, indx], color='blue', alpha=0.6, label='Actual', linestyle='--')

        # Plot original X_test values for x-axis
        plt.scatter(X_total_sorted[:, 0], Y_pred_sorted[:, indx], color='red', alpha=0.6)
        plt.scatter(X_total_sorted[:, 0], Y_total_sorted[:, indx], color='blue', alpha=0.6)

        # Labels and Title
        plt.xlabel(x_axis_feature)
        plt.ylabel(head_train[out_list[indx] - 1])
        plt.title('Actual Testing Data and Predicted Data vs Input')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'Input_vs_PredAct_{indx+1}.png', dpi=500)
        plt.show()

# Calculate R^2 for the scatter plot for each output variable
r2_scatter = []
for i in range(Y_pred.shape[1]):
    r2 = r2_score(Y_total[:, i], Y_pred[:, i])
    r2_scatter.append(r2)
    #print(f'R^2 for output {i+1} (scatter plot): {r2}')

# Plot scatter plots for each output variable in separate windows
for i in range(Y_pred.shape[1]):
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_total[:, i], Y_pred[:, i], color='red', alpha=0.6, label='Data')
    plt.plot([Y_total[:, i].min(), Y_total[:, i].max()], [Y_total[:, i].min(), Y_total[:, i].max()], color='blue', linestyle='-', label='x = y')  # Plot x=y line
    plt.title(f'Actual vs ANN Predicted Data ({head_train[out_list[i] - 1]})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Annotate R^2 value
    plt.text(0.05, 0.95, rf'$R^2= {r2_scatter[i]:.6f}$', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))

    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(f"Act_vs_Pred_{i+1}.png", dpi=500)
    plt.show()

