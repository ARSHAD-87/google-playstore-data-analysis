# <----------Importing Libraries---------->

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import os
nltk.download('vader_lexicon')
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# <------------Loading and reviewing Dataset---------->

apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')
print(apps_df.head())
print(reviews_df.head())


# <----------Data Cleaning---------->

# Handling missing values and duplicates
apps_df= apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df=apps_df[apps_df['Rating'] <= 5]
reviews_df.dropna(subset=['Translated_Review'], inplace=True)

# Convert the 'installs' column to numeric by removing the '+' and ',' characters
apps_df['Installs']=apps_df['Installs'].str.replace('+', '').str.replace(',', '').astype(int)

# Convert the 'Price' column to numeric by removing the '$' character
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)
merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')
merged_df.head()


# <----------Data Transmission---------->

# Convert the 'Size' column to numeric by handling 'M' and 'k' suffixes
def convert_size (size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) /1024
    else:
        return np.nan
apps_df['Size'] = apps_df['Size'].apply(convert_size)

# Convert the 'Reviews' column to integer
apps_df['Reviews']= apps_df['Reviews'].astype(int)

# Apply log transformation to 'Installs' and 'Reviews' columns
apps_df['Log_Installs'] = np.log1p(apps_df['Installs'])
apps_df['Log_Riviews'] = np.log1p(apps_df['Reviews'])

# Create a new column 'Rating_Group' based on the 'Rating' column
def rating_group(rating):
    if rating >= 4:
        return 'Top rated app'
    elif rating >= 3:
        return 'Above average'
    elif rating >= 2:
        return 'Average'
    else:
        return 'Below average'
apps_df['Rating_Group'] = apps_df['Rating'].apply(rating_group)

# Create a new column 'Revenue' by multiplying 'Installs' and 'Price'
apps_df['Revenue'] = apps_df['Installs'] * apps_df['Price']

# Sentiment Analysis on user reviews
sia = SentimentIntensityAnalyzer()
# Calculate sentiment scores for each review
reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Convert 'Last Updated' column to datetime format
apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'], errors='coerce')

# Extract year from 'Last Updated' column
apps_df['Year']=apps_df['Last Updated'].dt.year


# <----------Plotly Graphs---------->

# Create directory for HTML files if it doesn't exist
html_files_path="./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)
    
# Initialize a variable to hold all plot containers
plot_containers=""

# save each plotly figure to an html file
def save_plot_as_html(fig, a, b, filename, insight):
    global plot_containers
    file_path = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')

    # Append the plot and its insight to the plot_containers variable   
    plot_containers += f"""
    <div class="plot-container" 
         id="{filename}" 
         data-start="{a}" 
         data-end="{b}" 
         onclick="openPlot('{filename}')">
         
        <div class="plot">{html_content}</div>
        <div class="insight">{insight}</div>

        <script>
        (function() {{
            function displayPlotBasedOnTime(a, b, filename) {{
                var currentTime = new Date();
                var currentHour = currentTime.getHours();
                console.log('Received:', a, b, filename);

                a = parseInt(a, 10);
                b = parseInt(b, 10);
                console.log('Parsed:', a, b);

                var container = document.getElementById(filename);
                if (!container) return;

                if (currentHour >= a && currentHour < b) {{
                    console.log("Displaying plot");
                    container.style.display = "block";
                }} else {{
                    console.log("Hiding plot");
                    container.innerHTML = "<h3 style='color:white; text-align:center; padding-top:50%;'>" +
                    "This plot is available between " +
                    (a > 12 ? (a - 12) + ":00 PM" : (a == 12 ? "12:00 PM" : a + ":00 AM")) + " and " +
                    (b > 12 ? (b - 12) + ":00 PM" : (b == 12 ? "12:00 PM" : b + ":00 AM")) + " IST</h3>";
                }}
            }}

            document.addEventListener('DOMContentLoaded', function () {{
                var el = document.getElementById("{filename}");
                if (el) {{
                    displayPlotBasedOnTime(el.dataset.start, el.dataset.end, el.id);
                }}
            }});
        }})();
        </script>
    </div>
    """

    fig.write_html(file_path, full_html=False, include_plotlyjs='inline')
    
# Common plot settings
plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}

# Figure 1
category_counts=apps_df['Category'].value_counts().nlargest(10)
fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x':'Category', 'y':'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)

save_plot_as_html(fig1,"0","24","Category Graph 1.html","The top categories on the Play Store are dominated by tools, entertainment, and productivity apps")

# Figure 2
type_counts=apps_df['Type'].value_counts()
fig2=px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App types Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig2,"0","24","Type Graph 2.html","Most apps on the Playstore are free, indicating a strategy to attract users first and monetize through ads or inapp purchases.")

# Figure 3
fig3=px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=["#636EFA"],
    width=400,
    height=300
)
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,    
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig3,"0","24","Rating Graph 3.html","Ratings are skewed towards higher values, sugessting that most apps are favorable by users ")

# Figure 4
sentiment_counts=reviews_df['Sentiment_Score'].value_counts()
fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x':'Sentiment Score', 'y':'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig4,"0","24","Sentiment Graph 4.html","Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments")

# Figure 5
install_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5=px.bar(
    x=install_by_category.values,
    y=install_by_category.index,
    orientation='h',
    labels={'x':'Installs', 'y':'Category'},
    title='Installs by Category',
    color=install_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig5,"0","24","Installs Graph 5.html","The categoris with the most installs are social and communication apps, reflecting their broad appeal and daily usage")

# Figure 6
updates_per_year=apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6=px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x':'Year', 'y':'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FE'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig6,"0","24","Updates Graph 6.html", "Updades have been increasing over the years, indicating that developers are actively maintaining and improving their apps.")

# Figure 7
revenue_by_category=apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7=px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x':'Category', 'y':'Revenue'},
    title='Revenue by Category',
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)
fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig7,"0","24","Revenue Graph 7.html","Categories such as Family and Lifestyle lead in revenue generation, indicating their monetization potential.")

# Figure 8
genre_counts=apps_df['Genres'].str.split(';',expand=True).stack().value_counts().nlargest(10)
fig8=px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x':'Genre', 'y':'Count'},
    title='Top Genres',
    color=genre_counts.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)
fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig8,"0","24","Genre Graph 8.html","Action and Entertainment genres are the most common, reflecting users' prefrence for engaging and easy-to-play games.")

# Figure 9
fig9=px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=400,
    height=300
)
fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig9,"0","24","Update X Rating Graph 9.html","The Scatter plot shows a weak correlation between the last update and ratings, suggesting that more frequent updates don't always result in better ratings.")

# Figure 10
fig10=px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=400,
    height=300
)
fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig10,"0","24","Paid Free Graph 10.html","Paid apps generally have higher ratings compared to free apps,suggesting that users expect higher quality from apps they pay for.")

# Figure 11 
filter1_df=apps_df[
    (apps_df['Size'] >= 10) &
    (apps_df['Last Updated'].dt.month == 1)
]
avg_rating=filter1_df.groupby('Category')['Rating'].mean()
categories_to_keep=avg_rating[avg_rating >= 4.0].index
filter2_df=filter1_df[filter1_df['Category'].isin(categories_to_keep)]
top_categories=filter2_df.groupby('Category')['Installs'].sum().nlargest(10).index
final_df=filter2_df[filter2_df['Category'].isin(top_categories)]
chart_data = final_df.groupby('Category').agg(
        Average_Rating=('Rating', 'mean'),
        Total_Reviews=('Reviews', 'sum')
    ).reset_index()

chart_data = chart_data.sort_values('Average_Rating', ascending=False)

fig11 = make_subplots(specs=[[{"secondary_y": True}]])

fig11.add_trace(
        go.Bar(
            x=chart_data['Category'],
            y=chart_data['Average_Rating'],
            name='Average Rating',
            marker_color='rgb(26, 118, 255)',
            text=chart_data['Average_Rating'].round(2),
            textposition='auto',
        ),
        secondary_y=False,
    )

fig11.add_trace(
        go.Bar(
            x=chart_data['Category'],
            y=chart_data['Total_Reviews'],
            name='Total Reviews',
            marker_color='rgb(255, 127, 14)',
            text=chart_data['Total_Reviews'],
            texttemplate='%{text:.2s}', 
            textposition='auto',
        ),
        secondary_y=True,
    )

fig11.update_layout(
    title='Average Rating vs Total Reviews by Installs',
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(
        title="Average Rating (out of 5)",
        title_font=axis_font,
        range=[3.5, 5],
        gridcolor='gray'
    ),
    yaxis2=dict(
        title="Total Number of Reviews",     
        title_font=axis_font,
        overlaying='y',
        side='right',
        gridcolor='gray'
    ),
    margin=dict(l=10, r=11, t=30, b=10),
    width=400,
    height=300,
    legend=dict(
        orientation='h',         
        yanchor='bottom',
        y=-0.6,
        xanchor='center',
        x=0.5
    )
)
save_plot_as_html(fig11,"15","17","Average Rating vs Total reviews Graph 11.html","High review counts (popularity) don't always guarantee a perfect average rating, even for top-tier apps.")

# Figure 12
country_list_iso = [
    'USA', 'IND', 'CHN', 'BRA', 'RUS', 'GBR', 'DEU', 'FRA', 'JPN', 'CAN', 
    'AUS', 'MEX', 'IDN', 'PAK', 'NGA', 'BGD', 'EGY', 'VNM', 'TUR', 'IRN',
    'THA', 'ZAF', 'ITA', 'ESP', 'KOR', 'COL', 'ARG', 'POL', 'UKR', 'SAU'
]
apps_df_geo = apps_df.copy()
apps_df_geo['Country'] = np.random.choice(country_list_iso, len(apps_df_geo))


filtered_df = apps_df_geo[~apps_df_geo['Category'].str.startswith(('A', 'C', 'G', 'S'))]
top_5_cats_by_installs = filtered_df.groupby('Category')['Installs'].sum().nlargest(5).index

filtered_df = filtered_df[filtered_df['Category'].isin(top_5_cats_by_installs)]

map_data = filtered_df.groupby(['Country', 'Category'])['Installs'].sum().reset_index()
map_data = map_data[map_data['Installs'] > 1_000_000]

fig12 = px.choropleth(
    map_data,
    locations="Country",           
    locationmode="ISO-3",        
    color="Installs",              
    hover_name="Country",          
    hover_data={                   
        "Country": False,
        "Category": True,
        "Installs": ':,2s' 
    },
    animation_frame="Category",
    color_continuous_scale=px.colors.sequential.Plasma,
    scope="world",               
    title="Global Installs by Category (>1M, Filtered, Random Data)",
    width=plot_width,
    height=plot_height
)

fig12.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    margin=dict(l=10, r=10, t=30, b=10),
    geo=dict(
        bgcolor='black',
        lakecolor='black',
        landcolor='gray',
        subunitcolor='white'
    )
)
fig12.layout.updatemenus = None

save_plot_as_html(fig12, "18", "20", "Category Choropleth Graph 12.html", "'ENTERTAINMENT' app installs are highly concentrated in a few key countries, while 'EDUCATION' has a much wider global footprint.")

# Figure 13
apps_df['Android_Ver_Numeric'] = pd.to_numeric(
    apps_df['Android Ver'].str.extract(r'(\d+(?:\.\d+)?)', expand=False),
    errors='coerce'
)
top_3_categories = apps_df['Category'].value_counts().nlargest(3).index

free_apps = apps_df[
    (apps_df['Type'] == 'Free') &
    (apps_df['Installs'] >= 10000) &
    (apps_df['Android_Ver_Numeric'] > 4.0) &
    (apps_df['Size'] > 15) &
    (apps_df['Content Rating'] == 'Everyone') &
    (apps_df['App'].str.len() <= 30) &
    (apps_df['Category'].isin(top_3_categories))
]

paid_apps = apps_df[
    (apps_df['Type'] == 'Paid') &
    (apps_df['Installs'] >= 10000) &
    (apps_df['Revenue'] >= 10000) &
    (apps_df['Android_Ver_Numeric'] > 4.0) &
    (apps_df['Size'] > 15) &
    (apps_df['Content Rating'] == 'Everyone') &
    (apps_df['App'].str.len() <= 30) &
    (apps_df['Category'].isin(top_3_categories))
]

filtered_df = pd.concat([free_apps, paid_apps], ignore_index=True)
grouped_df = filtered_df.groupby(['Category', 'Type'])[['Installs', 'Revenue']].mean().reset_index()

fig13 = make_subplots(specs=[[{"secondary_y": True}]])
colors = {
    'Free': {'Installs': '#1f77b4', 'Revenue': '#d62728'},
    'Paid': {'Installs': '#aec7e8', 'Revenue': '#ff9896'}
}
for app_type in ['Free', 'Paid']:
    data = grouped_df[grouped_df['Type'] == app_type]

    fig13.add_trace(
        go.Bar(
            x=data['Category'],
            y=data['Installs'],
            name=f'Avg-Inst({app_type[:1]})',
            marker_color=colors[app_type]['Installs']
        ),
        secondary_y=False
    )

    fig13.add_trace(
        go.Bar(
            x=data['Category'],
            y=data['Revenue'],
            name=f'Avg-Rev({app_type[:1]})',
            marker_color=colors[app_type]['Revenue']
        ),
        secondary_y=True
    )

fig13.update_yaxes(
    title_text="Average Installs",
    secondary_y=False,
    title_font=axis_font,
    color=text_color,
    gridcolor='#444'
)
fig13.update_yaxes(
    title_text="Average Revenue ($)",
    secondary_y=True,
    title_font=axis_font,
    color=text_color,
    gridcolor='#444',
    overlaying='y',
    side='right'
)
fig13.update_xaxes(
    title_font=axis_font,
    color=text_color
)
fig13.update_layout(
    title='Avg Installs vs. Avg Revenue',
    xaxis_title='Top 3 Categories',
    barmode='group',
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    width=plot_width,
    height=plot_height,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        orientation='h',         
        yanchor='bottom',
        y=-0.6,
        xanchor='center',
        x=0.5
    )

)
save_plot_as_html(fig13,"13","14","Dual Axis Chart Graph 13.html","'GAME' revenue depends on high installs, but 'PRODUCTIVITY' apps can succeed with a high-price, niche-user model.")

# Figure 14
df_filtered = apps_df[
    (apps_df['Reviews'] > 500) &
    (~apps_df['App'].str.startswith(('x', 'y', 'z', 'X', 'Y', 'Z'))) &
    (~apps_df['App'].str.contains('s', case=False)) &
    (apps_df['Category'].str.startswith(('E', 'C', 'B')))
].copy()

translation_map = {
    'BEAUTY': 'सौंदर्य (Beauty)',
    'BUSINESS': 'வணிகம் (Business)',
    'DATING': 'Dating'
}
df_filtered['Category_Translated'] = df_filtered['Category'].apply(lambda x: translation_map.get(x, x))

df_agg = df_filtered.groupby(
    ['Category_Translated', pd.Grouper(key='Last Updated', freq='MS')]
)['Installs'].sum().reset_index() 

df_agg = df_agg.sort_values(by=['Category_Translated', 'Last Updated'])
df_agg['MoM_Growth_Pct'] = df_agg.groupby('Category_Translated')['Installs'].pct_change()

df_agg['Significant_Growth'] = df_agg['MoM_Growth_Pct'] > 0.20

fig14 = px.line(
    df_agg, 
    x='Last Updated', 
    y='Installs', 
    color='Category_Translated', 
    title='Monthly Installs Trend',
    labels={'Category_Translated': 'Category', 'Last Updated': 'Month', 'Installs': 'Total Installs'},
    width=plot_width,
    height=plot_height
)

shapes_list = []
growth_periods = df_agg[df_agg['Significant_Growth'] == True]

for index, row in growth_periods.iterrows():
    start_date = row['Last Updated']
    # Assuming monthly data, end date is one month after start date
    end_date = start_date + pd.DateOffset(months=1)        
    shapes_list.append(
        go.layout.Shape(
            type="rect", 
            xref="x",
            yref="paper", 
            x0=start_date,
            y0=0,
            x1=end_date,
            y1=1, 
            fillcolor="lightgreen",
            opacity=0.2,
            layer="below",
            line_width=0, 
        )
    )

fig14.update_layout(
    shapes=shapes_list,
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font,gridcolor='#444'),
    yaxis=dict(title_font=axis_font,gridcolor='#444'),
    legend=dict(font=dict(size=10)),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig14, "18", "21", "TimeSeries Graph 14.html","'BUSINESS' app growth is volatile and spiky, whereas 'ENTERTAINMENT' app growth is stable and more predictable.")

# Figure 15
avg_subjectivity_df = reviews_df.groupby('App')['Sentiment_Subjectivity'].mean().reset_index()

df_for_plot15 = pd.merge(apps_df, avg_subjectivity_df, on='App', how='inner')
categories_fig15 = ['GAME', 'BEAUTY', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'ENTERTAINMENT', 'SOCIAL', 'EVENTS']

df_filtered_15 = df_for_plot15[
    (df_for_plot15['Rating'] > 3.5) &
    (df_for_plot15['Category'].isin(categories_fig15)) &
    (df_for_plot15['Reviews'] > 500) &
    (~df_for_plot15['App'].str.contains('s', case=False)) &
    (df_for_plot15['Sentiment_Subjectivity'] > 0.5) & 
    (df_for_plot15['Installs'] > 50000) &
    (df_for_plot15['Size'].notna()) 
].copy()

translation_map_15 = {
    'BEAUTY': 'सौंदर्य (Beauty)',
    'BUSINESS': 'வணிகம் (Business)',
    'DATING': 'Dating (German)' 
}
df_filtered_15['Category_Translated'] = df_filtered_15['Category'].apply(
    lambda x: translation_map_15.get(x, x)
)

unique_categories = df_filtered_15['Category_Translated'].unique()
color_map_15 = {}
for cat in unique_categories:
    if cat == 'GAME': 
        color_map_15[cat] = 'pink'

fig15 = px.scatter(
    df_filtered_15,
    x='Size',
    y='Rating',
    size='Installs',
    color='Category_Translated',
    color_discrete_map=color_map_15,
    hover_name='App',
    hover_data=['Category', 'Installs', 'Size', 'Rating', 'Sentiment_Subjectivity'],
    title='App Size vs. Rating',
    labels={
        'Size': 'Size (MB)', 
        'Rating': 'Average Rating', 
        'Category_Translated': 'Category',
        'Installs': 'Total Installs'
    },
    size_max=50,
    width=plot_width,
    height=plot_height
)

fig15.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font, gridcolor='#444'),
    yaxis=dict(title_font=axis_font, gridcolor='#444'),
    legend=dict(font=dict(size=9)),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig15, "17", "19", "Bubble Chart Graph 15.html", "For popular apps, users clearly do not care about large file sizes as long as the quality (rating) is high.")

# Figure16
df_filtered_16 = apps_df[
    (apps_df['Rating'] >= 4.2) &
    (~apps_df['App'].str.contains(r'\d')) & 
    (apps_df['Category'].str.startswith(('T', 'P'))) &
    (apps_df['Reviews'] > 1000) &
    (apps_df['Size'].between(20, 80)) 
].copy()

translation_map_16 = {
    'TRAVEL_AND_LOCAL': 'Voyage et local (Travel & Local)',   # French
    'PRODUCTIVITY': 'Productividad (Productivity)',           # Spanish
    'PHOTOGRAPHY': '写真 (Photography)'                        # Japanese
}
df_filtered_16['Category_Translated'] = df_filtered_16['Category'].apply(
    lambda x: translation_map_16.get(x, x)
)

df_monthly_16 = df_filtered_16.groupby(
    [pd.Grouper(key='Last Updated', freq='MS'), 'Category_Translated']
)['Installs'].sum().reset_index()
df_monthly_16 = df_monthly_16.sort_values(by=['Category_Translated', 'Last Updated'])

df_monthly_16['MoM_Growth_Pct'] = df_monthly_16.groupby('Category_Translated')['Installs'].pct_change()
high_growth_months = df_monthly_16[df_monthly_16['MoM_Growth_Pct'] > 0.25]['Last Updated'].unique()

df_cumulative_16 = df_monthly_16.copy()
df_cumulative_16['Cumulative_Installs'] = df_cumulative_16.groupby('Category_Translated')['Installs'].cumsum()

fig16 = px.area(
    df_cumulative_16,
    x='Last Updated',
    y='Cumulative_Installs',
    color='Category_Translated',
    title='Cumulative Installs Over Time',
    labels={
        'Last Updated': 'Month',
        'Cumulative_Installs': 'Cumulative Installs',
        'Category_Translated': 'Category'
    },
    width=plot_width,
    height=plot_height
)

shapes_list_16 = []
for month_start in high_growth_months:
    month_end = month_start + pd.DateOffset(months=1)
    shapes_list_16.append(
        go.layout.Shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=month_start,
            y0=0,
            x1=month_end,
            y1=1,
            fillcolor="yellow",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    )

fig16.update_layout(
    shapes=shapes_list_16,
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font, gridcolor='#444'),
    yaxis=dict(title_font=axis_font, gridcolor='#444'),
    legend=dict(font=dict(size=9)),
    margin=dict(l=10, r=10, t=30, b=10)
)

save_plot_as_html(fig16, "16", "18", "Stacked Area Graph 16.html", "'PHOTOGRAPHY' is the established market leader in installs, but 'PRODUCTIVITY' is the high-velocity challenger closing the gap.")

# Split the plot containers into individual plots for display
plot_containers_split=plot_containers.split('</div>')

# Wrap each plot container in a div for better styling and interaction
if len(plot_containers_split) > 1:
    final_plot=plot_containers_split[-2] + '</div>'
else:
    final_plot=plot_containers
    

# <----------Dashboard Creation----------->

# Create the final HTML dashboard
dashboard_html="""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Google Play Store Review Analysis</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        background-color: #333;
        color: #fff;
        margin: 0;
        padding: 0;
        }}
    .header {{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background-color: #444;
        }}
    .header img {{
        margin: 0 10px;
        height: 50px;
        }}
    .container {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        padding: 20px;
        }}
    .plot-container {{
        border: 2px solid #555;
        margin: 10px;
        padding: 10px;
        width: {plot_width}px;
        height: {plot_height}px;
        overflow: hidden;
        position: relative;
        cursor: pointer;
        }}
    .insight {{
        display: none;
        position: absolute;
        right: 10px;
        top: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 5px;
        border-radius: 5px;
        color: #fff;
        }}
    .plot-container:hover .insight {{
        display: block;
        }}
</style>
<script>
    function openPlot(filename) {{
        window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo" />
      <h1>Google Play Store Review Analysis</h1>
      <img src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Google_Play_2022_logo.svg" alt="Google Play Store Logo" />
    </div>
    <div class="container">
        {plots}       
    </div>
</body>
</html>
"""
# Combine all plot containers into the final HTML
final_html=dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height)

# Save the final HTML to a file
dashboard_path=os.path.join(html_files_path,"index.html")

# Write the final HTML to a file
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(final_html)
  
# Open the dashboard in the default web browser  
webbrowser.open('file://' + os.path.realpath(dashboard_path))