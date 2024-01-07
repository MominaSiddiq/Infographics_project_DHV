
"""
Created on Sat Dec 30 01:45:50 2023

@author: macbook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_and_clean_data(filename):
    """
    This function reads data into a pandas dataframe take transpose and 
    clean the data.

    Parameters
    ----------
    filename : csv
       take the filenames as an argument.

    Returns
    -------
    two dataframes.

    """

    # read the data from the file
    data = pd.read_csv(filename, skiprows = 4, index_col = False)

    # cleaning the data(removing empty rows etc.)
    data.dropna(axis = 0, how = 'all', inplace = True)

    # remove emty columns
    data.dropna(axis = 1, how = 'all', inplace = True)

    # Drop the unnecessary columns in the data
    data.drop(columns = ['Country Code', 'Indicator Name',
              'Indicator Code'], inplace = True)

    # taking the transpose
    years_col_df = data.T

    # setting the header
    years_col_df.columns = years_col_df.iloc[0]
    years_col_df = years_col_df[1:]

    # reset index for making years as columns
    years_col_df = years_col_df.reset_index()
    years_col_df = years_col_df.rename(columns = {'index': 'Year'})

    # setting years as index
    years_col_df.set_index('Year', inplace = True)

    # removing empty rows
    years_col_df.dropna(axis = 0, how = 'all', inplace = True)

    # removing empty columns
    years_col_df.dropna(axis = 1, how = 'all', inplace = True)

    # Removeing any unnecessary columns in the transpose of data
    years_col_df = years_col_df.loc[:, ~years_col_df.columns.duplicated()]

    # Removing any duplicated rows
    years_col_df = years_col_df[~years_col_df.index.duplicated(keep = 'first')]

    # taking the transpose again for country column
    country_col_df = years_col_df.T

    # Reset index for making countries as columns
    country_col_df = country_col_df.reset_index().rename(
        columns={'index': 'Country'})

    return country_col_df, years_col_df


def filtered_data(df, countries, start_year, end_year):
    """
    filtering data on selective years and countries for all the indicators 

    Parameters
    ----------
    data : python dataframe

    Returns
    -------
    filtered data.

    """
    # Ensure the DataFrame has an index named 'Country' or reset it if necessary
    if df.index.name != 'Country Name':
        if 'Country Name' in df.columns:
            df.set_index('Country Name', inplace = True)
        else:
            print("Country Name column not found.")
            return None

    # Filter for the selected countries
    filtered_df = df[df.index.isin(countries)]

    # Convert years to string if necessary
    start_year, end_year = str(start_year), str(end_year)

    # Filter for the range of years, ensuring they are in string format
    filtered_df = filtered_df.loc[:, start_year:end_year]

    return filtered_df


def group_data_into_periods(data, start_year, end_year, period_length):
    """
    Group data into specified periods.

    Parameters:
    ----------
    data : DataFrame
        DataFrame containing the data to group.

    start_year : int
        The starting year of the data.

    end_year : int
        The ending year of the data.

    period_length : int
        The length of each period in years.

    Returns:
    -------
    DataFrame
        DataFrame with grouped data.
    """
    # Transpose the DataFrame to have years as rows
    df_transposed = data.transpose()

    # Ensure the DataFrame contains the specified range of years
    df_transposed = df_transposed.loc[str(start_year):str(end_year)]

    # Convert the index to integer for easier comparison
    df_transposed.index = df_transposed.index.astype(int)

    # Group data by periods
    grouped_data = {}
    for year in range(start_year, end_year + 1, period_length):
        period_end = min(year + period_length - 1, end_year)
        period_label = f'{year}-{period_end}'
        period_data = df_transposed.loc[year:period_end].sum()
        grouped_data[period_label] = period_data

    return pd.DataFrame(grouped_data)


def plot_area_graph_for_emissions(ghg_df, ax):
    """
    Plot an area graph for greenhouse gas emissions from the filtered_dfs dictionary.

    Parameters:
    ----------
    ghg_dfs : greenhouse gas df 
         filtered dataframes of ghg emission.

    ax : matplotlib.axes.Axes
        The axes on which to plot the graph.

    Returns
    -------
    None.
    """
    # Check if 'Country Name' is the index or a column, and set it as the index if necessary
    if 'Country Name' in ghg_df.columns:
        ghg_df = ghg_df.set_index('Country Name')

    elif ghg_df.index.name != 'Country Name' and 'Country Name' not in ghg_df.index:
        raise KeyError(
            "The DataFrame does not have 'Country Name' as a column or index name.")

    # Transpose the dataframe to get years as the index
    ghg_transposed = ghg_df.transpose()
    ghg_transposed.index = ghg_transposed.index.map(int)
    total_emissions_by_year = ghg_transposed.sum(axis = 1)

    # Plot the total emissions over time using the passed 'ax' for plotting
    ax.plot(total_emissions_by_year.index, total_emissions_by_year,
            color = '#c27ba0', linewidth = 3, label = 'Total Emissions')

    # Annotate peak points
    peaks = total_emissions_by_year[(total_emissions_by_year.shift(1) < total_emissions_by_year) &
                                    (total_emissions_by_year.shift(-1) < total_emissions_by_year)]
    for year, value in peaks.items():
        ax.annotate(f'{value:.2f}', xy = (year, value), xytext = (0, 20),
                    textcoords = 'offset points', ha = 'center', weight = 'bold', size = 14,
                    arrowprops=dict(facecolor = 'black', arrowstyle = '->'),
                    bbox=dict(boxstyle = "round,pad=0.3", fc = "lightgray", ec = "black", alpha = 0.5))
 
    # Set titles and labels with specified font sizes and weights
    ax.set_title('Greenhouse Gases Emission', fontsize = 20, fontweight = 'bold')

    # Configure the legend for this specific plot if needed
    ax.legend(loc = 'upper left')

    # Customize the x-axis ticks for better readability
    ax.tick_params(axis = 'x', rotation = 45)

    # Making grids
    ax.grid(linestyle = '--', linewidth = 0.5, color = 'gray')


def plot_renewable_energy_consumption(renewable_df, ax):
    """
    Plot a horizontal bar graph for renewable energy consumption with a gap of 5 years.

    Parameters:
    ----------
    renewable_df : daatframe
        Renewable energy consumption dataframe.

    ax : matplotlib.axes.Axes
        The axes on which to plot the graph.

    Returns
    -------
    None.
    """
    # Check if 'Country Name' is a column; if so, set it as the index
    if 'Country Name' in renewable_df.columns:
        renewable_df = renewable_df.set_index('Country Name')

    elif renewable_df.index.name != 'Country Name':
        raise KeyError(
            "The DataFrame does not have 'Country Name' as a column or index name.")

    # Transpose the dataframe to have countries as columns and years as rows
    renewable_transposed = renewable_df.transpose()
    renewable_transposed.index = renewable_transposed.index.map(int)

    # Select only the years that are multiples of 5
    years_to_plot = [
        year for year in renewable_transposed.index if year % 10 == 0]
    renewable_filtered = renewable_transposed[renewable_transposed.index.isin(
        years_to_plot)]
    total_consumption_by_year = renewable_filtered.sum(axis = 1)

    # Use the passed 'ax' for plotting
    bars = ax.barh(total_consumption_by_year.index,
                   total_consumption_by_year, color = '#088193', height = 3.0)
    bbox_props = dict(boxstyle = "square,pad=0.5",
                      fc = "lightgrey", alpha = 0.5, lw = 0)

    # Annotate each bar with the year and the value
    for bar in bars:
        year = bar.get_y() + bar.get_height() / 2
        value = bar.get_width()
        ax.text(value, year, f'{value:.2f} %', va = 'center', ha = 'left',
                color = 'black', weight = 'bold', size = 14, bbox = bbox_props)

    # Customize the plot
    ax.set_title('Total Renewable Energy Consumption Over Time',
                 fontsize = 20, fontweight = 'bold')

    # Set the y-axis ticks to the filtered years
    ax.set_yticks(years_to_plot)
    ax.set_yticklabels(years_to_plot)


def plot_urban_rural_population(urban_data, rural_data, axs):
    """
    Plots multiple pie charts for urban vs rural population over different periods onto given axes.

    Parameters:
    ----------
    urban_data : DataFrame
        DataFrame with periods as columns and aggregated urban population as values.

    rural_data : DataFrame
        DataFrame with periods as columns and aggregated rural population as values.

    axs : list of matplotlib.axes.Axes
        The list of axes on which to plot the graphs.

    Returns
    -------
    None.    
    """
    colors = ['#088193', '#85a6a9']  # Blue for urban, Light gray for rural
    num_periods = len(urban_data.columns)
    assert len(
        axs) >= num_periods, "The number of axes provided must match the number of periods."

    for i, period in enumerate(urban_data.columns):
        period_urban_pop = urban_data[period].sum()
        period_rural_pop = rural_data[period].sum()

        data = [period_urban_pop, period_rural_pop]
        explode = (0.1, 0)

        # Plot the pie chart on the specified Axes
        wedges, texts, autotexts = axs[i].pie(data, explode=explode,
                                              colors = colors,
                                              autopct = '%1.1f%%', 
                                              startangle = 90, 
                                              pctdistance = 0.85)

        for autotext in autotexts:
            autotext.set_size(16)
            autotext.set_color('black')
            autotext.set_weight('bold')

        centre_circle = plt.Circle((0, 0), 0.70, fc = 'white')
        axs[i].add_artist(centre_circle)

        axs[i].set_title(f"{period}", color = 'black',
                         weight = 'bold', fontsize = 16)
        axs[i].set_aspect('equal')

    # Return the wedge objects from the last pie chart to use for creating a legend
    return wedges


def plot_agriculture_land(data, ax):
    """
    Creates a stacked bar chart for grouped data on a specific subplot.

    Parameters:
    ----------
    data : DataFrame
        DataFrame containing the agriculture data to plot, indexed by period with categories as columns.

    ax : matplotlib.axes.Axes
        The axes on which to plot the graph.

    Returns
    -------
    None.  
    """
    # Initialize the bottom position for the stacked bars
    bottom = np.zeros(len(data))

    colors = ['#85a6a9', '#1aa3a6', '#c27ba0', '#759e5b']

    # Create the stacked bar chart
    for idx, (column, color) in enumerate(zip(data.columns, colors)):
        ax.bar(data.index, data[column], bottom = bottom,
               color = color, label = column, width = 0.6)
        bottom += data[column].values

    # Annotate the bars with the total for each period
    for idx, period in enumerate(data.index):
        total = bottom[idx]
        ax.text(idx, total, f'{total:.1f}', va = 'bottom',
                ha = 'center', color = 'black', weight = 'bold', size = 14)

    ax.set_title('Agricultural Land Use (1990 - 2020)',
                 fontsize = 20, fontweight = 'bold')

    # Add the legend
    ax.legend(title = 'Period', fontsize = 12, frameon = False)

    # Set the x-axis labels
    ax.set_xticks(range(len(data.index)))
    ax.set_xticklabels(data.index, rotation = 45)


def plot_temperature_trends(data, ax):
    """
    Plots a line graph for temperature trends across various countries on a specific subplot.

    Parameters:
    ----------
    data : DataFrame 
        Containing temperature data for various countries across years.

    ax : matplotlib.axes.Axes
        The axes on which to plot the graph.

    Returns
    -------
    None.  
    """
    color_list = ['#85a6a9', '#1aa3a6', '#c27ba0', '#759e5b']

    # Plotting the data
    for i, country in enumerate(data.columns):
        # Cycle through colors if there are more countries than colors
        color = color_list[i % len(color_list)]
        ax.plot(data.index, data[country],
                label = country, color = color, linewidth = 2)

    ax.set_title('Annual Surface Temperature Anomalies (°C)',
                 fontsize = 20, fontweight = 'bold')
    
    ax.tick_params(axis = 'x', rotation = 45)
    ax.legend(title = 'Period', fontsize = 12, frameon = False)


if __name__ == "__main__":

    # List of files
    filename = ["greenhouse_gas_emission.csv",
                "renewable_energy_consumption.csv", "urban_population.csv",
                "rural_population.csv", "agriculture_land.csv", "annual_surface_temperature.csv"]

    # Process each file and save the transposed data
    transposed_files = []

    for file in filename:
        # Process the data
        country_col_df, years_col_df = read_and_clean_data(file)

        # Create a new filename for the transposed data
        transposed_filename = file.replace('.csv', '_trans.csv')
        transposed_files.append(transposed_filename)

        # Save the transposed data
        country_col_df.to_csv(transposed_filename, index=False)

    # selecting countries
    selected_countries = ['Qatar', 'China', 'Pakistan',
                          'Netherlands', 'Portugal', 'United States', 'Canada', 'Bangladesh',
                          'Italy', 'Japan', 'Sri Lanka', 'New Zealand', 'Oman', 'United Kingdom']

    # selecting years
    start_year = 1990
    end_year = 2020

    # List to store filtered DataFrames
    filtered_dfs = {}

    # Read and filter each transposed file
    for transposed_file in transposed_files:
        # Read the transposed data
        df = pd.read_csv(transposed_file)

        # Filter the data
        filtered_df = filtered_data(
            df, selected_countries, start_year, end_year)

        # Add the filtered DataFrame to the list
        filtered_dfs[transposed_file] = filtered_df

        # Add the filtered DataFrame to the dictionary
        if filtered_df is not None:
            filtered_dfs[transposed_file] = filtered_df

    # Creating a dashboard
    # Define the number of rows needed for the subplots
    num_rows = 5
    num_columns = 4  # Assuming 4 pie charts in one row
    dpi = 300
    fig, axs = plt.subplots(5, 1, figsize=(30, 36), dpi=dpi)

    # Define color scheme for the dashboard
    background_color = '#F9F9F9'
    fig.patch.set_facecolor(background_color)

    # Plotting for greenhouse gas emissions
    ghg_df = filtered_dfs['greenhouse_gas_emission_trans.csv']
    plot_area_graph_for_emissions(ghg_df, axs[0])
    axs[0].set_facecolor('none')

    # Plotting for renewable energy consumption
    renewable_df = filtered_dfs['renewable_energy_consumption_trans.csv']
    plot_renewable_energy_consumption(renewable_df, axs[1])
    axs[1].set_facecolor('none')

    urban_df = filtered_dfs['urban_population_trans.csv']
    rural_df = filtered_dfs['rural_population_trans.csv']
    agriculture_df = filtered_dfs['agriculture_land_trans.csv']
    temperature_df = filtered_dfs['annual_surface_temperature_trans.csv']

    # Grouping in periods
    group_urban_pop = group_data_into_periods(
        urban_df, start_year, end_year, 10)
    group_rural_pop = group_data_into_periods(
        rural_df, start_year, end_year, 10)
    group_temp_data = group_data_into_periods(
        temperature_df, start_year, end_year, 10)
    agricultural_land_grouped = group_data_into_periods(
        agriculture_df, start_year, end_year, 10)

    # Plotting for urban vs rural population pie charts in the third row
    pie_axs = [plt.subplot2grid(
        (num_rows, num_columns), (2, i), fig=fig) for i in range(num_columns)]
    wedges = plot_urban_rural_population(
        group_urban_pop, group_rural_pop, pie_axs)

    for ax in pie_axs:
        ax.set_facecolor('#F9F9F9')

    # Add a title for the pie charts
    fig.text(0.5, axs[2].get_position().y0 + 0.185, 'Urban vs Rural Population Distribution',
             ha = 'center', va = 'center', size = 20, weight = 'bold', transform = fig.transFigure)

    # Get the legend handles and labels from one of the pie charts
    labels = ['Urban Population', 'Rural Population']
    pie_axs[2].legend(wedges, labels, loc='upper center', bbox_to_anchor = (-0.4, 0.02),
                      ncol = 2, fontsize = 16, frameon = False)

    # Plotting for agricultural land use
    plot_agriculture_land(agricultural_land_grouped, axs[3])
    axs[3].set_facecolor('none')

    # Plotting for annual surface temperature anomalies
    plot_temperature_trends(group_temp_data, axs[4])
    axs[4].set_facecolor('none')

    # Adjust layout
    plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.13 +
                        0.05, top = 0.92, hspace=0.5, wspace = 0.05)

    # Set the main dashboard title
    fig.suptitle('Climate Change and Environmental Impacts (1990 - 2020)',
                 fontsize = 34, color = 'navy', fontweight = 'bold', ha = 'center', y = 0.98)

    # Add student name and ID
    fig.text(0.5, 0.96, "Name: Momna Hammad | ID: 22047438",
             ha = 'center', fontweight = 'bold', va = 'top', fontsize = 22)

    font_style = {
        'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 24
    }

    # Define the explanation text
    explanation_text = (
        "This dashboard presents interrelated aspects of environmental change from 1990 to 2020.\n"
        "-Greenhouse gases emission escalated to a peak of over 2.2 million metric tons, signaling a critical environmental stressor linked to\n global warming.\n"
        "-A positive trend with a notable increase to 33.3% in recent years, reflecting a shift towards sustainable energy sources, a positive sign\n amidst growing environmental concerns.\n"
        "-The urban vs. rural population mark a stark increase in urbanization, with urban dwellers increasing from 42.7% to 63.8%,\n highlighting the migration patterns and the potential stress on urban ecosystems.\n"
        "-Agricultural land use data reflect shifts in land management, possibly adapting to demographic and climatic changes.\n"
        "-The surface temperature fluctuations with notable extremes in recent years, underscoring the irregular patterns that\n characterize climate volatility."
    )

    # Define properties for the bounding box
    bbox_props = dict(boxstyle = "square,pad=0.3",
                      fc = "#F0F0F0", ec = "black", lw = 2)

    # Add the wrapped text at the bottom of the figure
    fig.text(
        0.125, 0.01 + 0.05/2,
        explanation_text,
        ha = 'left',
        fontdict = font_style,
        bbox=bbox_props
    )
    # Show the figure 
    plt.show()
