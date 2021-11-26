import numpy as np
import pandas as pd
import os
import subprocess

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go


from wordcloud import WordCloud, STOPWORDS

class Plotter():
    def __init__(self):
        self.data = pd.read_csv('summary_anno.csv', sep='\t', index_col=0)
        self.tags = ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace', 'tenderness', 'transcendence']
        self.quads = {'Q1': '1','Q2': '2','Q3': '3','Q4': '4'}
        # images
 
    def get_word_cloud(self, string):
        stopwords = set(STOPWORDS)
        stopwords.add('emotion')
        stopwords.add('transmitted')
        stopwords.add('el')
        stopwords.add('un')
        stopwords.add('hi ha')
        stopwords.add('y')


        cloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=100,
                          min_font_size=20,
                          width=1500,
                          height=200)

        cloud.generate(string)
        return cloud

    def make_song_info(self, idx, df):
        df_subset = df.iloc[idx]
        spoti_link_emb = 'https://open.spotify.com/embed/track/{}'.format(df.loc[idx, 'track_id'])
        spoti_link = 'https://open.spotify.com/track/{}'.format(df.loc[idx, 'track_id'])
        muziek_link = 'https://www.muziekweb.nl/Embed/{}'.format(df.loc[idx, 'cdr_track_num'])

        # figure quads
        fig_quads = go.Figure()
        fig_quads.add_trace(go.Bar(x=df_subset[list(self.quads.values())], y=list(self.quads.keys()), orientation='h'))
        fig_quads.update_layout(title='Quadrant frequency')

        # figure moods
        fig_moods = go.Figure()
        fig_moods.add_trace(go.Barpolar(r=df_subset[self.tags], theta=self.tags))

        fig_moods.update_layout(title='Mood frequency',
            polar=dict(angularaxis=dict(showline=False), radialaxis=dict(visible = False)),
            font_size=11)

        # figure wordcloud
        try:
            txt_cld = df.iloc[idx].txt_free + ' ' + df.iloc[idx].txt_quad + ' ' + df.iloc[idx].txt_mood
            cloud = self.get_word_cloud(txt_cld)
        except:
            cloud = np.zeros((500,500))
        fig_cld = go.Figure(go.Image(z=cloud))
        fig_cld.update_layout(title='Wordcloud')
        fig_cld.update_xaxes(visible=False)
        fig_cld.update_yaxes(visible=False)

        layout = html.Div([
            html.Embed(src=muziek_link, height=40, width=350, style={'marginRight': 50, 'marginLeft': 50}),

            html.Embed(src=spoti_link_emb, height=80, width=300, style={'marginRight': 50, 'marginLeft': 50}),
            # html.H6('Summary:'),
            html.P('Summary: {0:.0f} annotators - preference ({1:.0f}%) - familiarity ({2:.0f}%) - tempo ({3:.1f} BPM)'.format(df.iloc[idx].num_users, 
                                                                                                                      float(df.iloc[idx].pref) * 100,
                                                                                                                      float(df.iloc[idx].fam) * 100, 
                                                                                                                      df.iloc[idx].tempo),
                   style={'text-align': 'center'}),
            html.Div(children=[            

            html.Div(children=[
                dcc.Graph(figure=fig_quads,  style={'height': '35vh'}),
            ], className='four columns'),

            html.Div(children=[
                dcc.Graph(figure=fig_moods,  style={'height': '35vh'}),
            ], className='eight columns'),
            ], className='twelve columns'),

            html.Div(children=[
                dcc.Graph(figure=fig_cld, style={'height': '30vh'}),
            ], className='twelve columns'),
            # 
        ], className='twelve columns')

        return layout


    def create_layout(self, app):
        with open('intro_txt.md', 'r') as file:
            intro_txt = file.read()
        layout = html.Div(style={'background-color':'#ffffff'}, children=[
            # header
            html.Div(className="row header",
                style={"background-color": "#f9f9f9",
                       "margin": "5px 5px 5px 5px"},
                children=[
                html.A(                
                    html.Img(src=app.get_asset_url('mtg.png'), alt='mtg_logo', height=70),
                    href='https://www.upf.edu/web/mtg/',
                    target='_blank',
                    className='three columns'),

                html.H3('MTG-MER: an open data set for personalized Music Emotion Recognition',
                        style={'text-align': 'right'},
                        className='nine columns'),
                ]),


            # text and graph
            html.Section(className='row', style={'padding':'0px'}, children=[
                html.Div(children=[
                    dcc.Markdown(intro_txt),

                html.Div(className='row', children=[

                    html.Div(children=[
                        dcc.Graph(id='graph-arousal-valence',  style={'height': '68vh'}),
                    ], className='eight columns'), 


                html.Div(children=[

                    html.Div(children=['Select AV representation:',
                            dcc.Dropdown(
                                id='dropdown-arousalvalence',
                                placeholder='Select AV representation:',
                                searchable=False,
                                clearable=False,
                                options=[
                                    {'label': 'Spotify API', 'value':'spoti_api'},
                                    {'label': 'Spotify  API + Standarization', 'value':'norm'},
                                    {'label': 'TROMPA participants', 'value':'users'},
                                ],
                                value='spoti_api',
                            ),
                        ],
                    ),

                    html.Div(children=['Colorize using Spotify API:',
                            dcc.Dropdown(
                                id='dropdown-color',
                                placeholder='Colorize using Spotify API:',
                                searchable=False,
                                clearable=False,
                                options=[
                                    {'label': 'None', 'value':'none'},
                                    {'label': 'Danceability', 'value':'danceability'},
                                    {'label': 'Popularity', 'value':'popularity'},
                                    {'label': 'Tempo', 'value':'tempo'},
                                    {'label': 'Loudness', 'value':'loudness'},
                                    {'label': 'Acousticness', 'value':'acousticness'},
                                    {'label': 'Speechiness', 'value':'speechiness'},
                                    {'label': 'Liveness', 'value':'liveness'},
                                ],
                                value='none',
                            ),
                        ],
                    ),

                    html.Div(
                        style={"margin": "20px 5px 20px 0px"},
                        children=["Mode",
                            dcc.Slider(
                                id="slider-mode",
                                min=-1,
                                max=1,
                                step=None,
                                value=-1,
                                marks={-1: 'All',
                                       0: 'Minor',
                                       1: 'Major'},
                                vertical=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"margin": "25px 5px 30px 0px"},
                        children=["Tempo",  
                            dcc.RangeSlider(
                                id="slider-tempo",
                                min=0,
                                max=220,
                                step=1,
                                value=[0, 220],
                                tooltip={"placement": "bottom", "always_visible": True},
                                allowCross=False,
                            ),
                        ],
                    ),

                    html.Div(
                        style={"margin": "25px 5px 30px 0px"},
                        children=["Key",  
                            dcc.Slider(
                                id="slider-key",
                                min=0,
                                max=15,
                                step=None,
                                value=15,
                                marks={0: 'C',
                                       1: 'C#',
                                       2: 'D',
                                       3: 'D#',
                                       4: 'E',
                                       5: 'F',
                                       6: 'F#',
                                       7: 'G',
                                       8: 'G#',
                                       9: 'A',
                                       10: 'A#',
                                       11: 'B',
                                       15: 'All'},

                                vertical=False,
                            ),
                        ],
                    ),

                    html.Div(
                        style={"margin": "25px 5px 30px 0px"},
                        children=["Num. Annotators",  
                            dcc.RangeSlider(
                                id="slider-users",
                                min=0,
                                max=np.max(self.data.num_users),
                                step=1,
                                value=[0, np.max(self.data.num_users)],
                                tooltip={"placement": "bottom", "always_visible": True},
                                vertical=False,
                            ),
                        ],
                    ),

                ], className='four columns'),
                ]),
                    
            ], className='six columns',
            ),

            # annotation info
            html.Div(
                [
                    html.Table(
                        id="table-element",
                        className="table__container",
                    )
                ],
                id="click-information",
                className='six columns',
            ),
            ]),
            
            # footer
            html.Div(className="row footer",
                style={"background-color": "#f9f9f9"},
                children=[

                html.P('Created by Juan Sebastián Gómez-Cañón',
                        style={'text-align': 'center'},),
                html.Div(
                    children=[
                    html.A(                
                        html.Img(src=app.get_asset_url('juan_gomez.png'), alt='juan_logo', height=25),
                        href='https://juansgomez87.github.io/',
                        target='_blank',
                        style={"margin": "0px 30px 0px 30px"}),
                     html.A(                
                        html.Img(src=app.get_asset_url('twitter.png'), alt='twitter', height=25),
                        href='https://twitter.com/juan_s_gomez',
                        target='_blank',
                        style={"margin": "0px 30px 0px 30px"}),
                    html.A(                
                        html.Img(src=app.get_asset_url('github.png'), alt='github', height=25),
                        href='https://github.com/juansgomez87',
                        target='_blank',
                        style={"margin": "0px 30px 0px 30px"}),               
                    html.A(                
                        html.Img(src=app.get_asset_url('scholar.png'), alt='scholar', height=25),
                        href='https://scholar.google.com/citations?user=IvIQqUwAAAAJ&hl=en',
                        target='_blank',
                        style={"margin": "0px 30px 0px 30px"}),               
                    ],
                    style={'text-align': 'center'},),
                ]),

        ])
        return layout



    def run_callbacks(self, app):
        @app.callback(
            Output("graph-arousal-valence", "figure"),
            [
                Input("dropdown-arousalvalence", "value"),
                Input("dropdown-color", "value"),
                Input("slider-mode", "value"),
                Input("slider-key", "value"),
                Input("slider-tempo", "value"),
                Input("slider-users", "value"),
                
            ],
        )
        def display_plot(av_rep, spoti_filt, sl_mode, sl_key, sl_tempo, sl_users):
            this_df = self.data
            list_opacity = (this_df.num_users / np.max(this_df.num_users)) ** 0.18

            # filter with musical properties
            if sl_mode != -1:
                this_df = this_df[this_df['mode'] == sl_mode].reset_index()
            if sl_key != 15:
                this_df = this_df[this_df['key'] == sl_key].reset_index()
            if sl_tempo != [0, 220]:
                this_df = this_df[(this_df['tempo'] >= sl_tempo[0]) & (this_df['tempo'] <= sl_tempo[1])].reset_index()
            if sl_users != [0, np.max(self.data.num_users)]:
                this_df = this_df[(this_df['num_users'] >= sl_users[0]) & (this_df['num_users'] <= sl_users[1])].reset_index()
            # show arousal valence representations
            if av_rep == 'spoti_api':
                aro_col = 'energy'
                val_col = 'valence'
                title = 'Spotify API'
                y_line = 0.5
                x_line = 0.5
            elif av_rep == 'norm':
                aro_col = 'norm_energy'
                val_col = 'norm_valence'
                title = 'Standarization'
                y_line = 0.0
                x_line = 0.0
            elif av_rep == 'users':
                # TODO!!!
                aro_col = 'energy'
                val_col = 'valence'
                title = 'Users (TODO)'
                y_line = 0.5
                x_line = 0.5

            if spoti_filt == 'none':
                this_color = None
                show_scale = False
            else:
                this_color = this_df[spoti_filt]
                show_scale = True
                list_opacity = np.ones(list_opacity.shape)

            axes = dict(title="Test", showgrid=True, zeroline=True, showticklabels=False)

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis=axes, yaxis=axes),
                legend=dict(x=-.1, y=1.2, orientation="h"),
            )

            
            data = []
            for sel, group in this_df.groupby('selection'):
                scatter = go.Scattergl(
                        x=group[val_col],
                        y=group[aro_col],
                        mode='markers',
                        name=sel,
                        marker=dict(size=6, symbol='circle', opacity=list_opacity[group.index], showscale=show_scale, line_width=1),
                        marker_color=this_color,
                        text=group['track_name'],
                    )
                data.append(scatter)

            figure = go.Figure(data=data, layout=layout)
            figure.update_layout(title=title)
            figure.update_xaxes(title_text='Valence')
            figure.update_yaxes(title_text='Arousal')
            figure.add_hline(y=y_line)
            figure.add_vline(x=x_line)            

            return figure

        @app.callback(
            Output('click-information', 'children'),
            [
                Input('graph-arousal-valence', 'clickData'),
                Input("dropdown-arousalvalence", "value"),
                Input("dropdown-color", "value"),
                Input("slider-mode", "value"),
                Input("slider-key", "value"),
                Input("slider-tempo", "value"),
            ],
        )
        def display_info(click_data, av_rep, spoti_filt, sl_mode, sl_key, sl_tempo):
            this_df = self.data

            # filter with musical properties
            if sl_mode != -1:
                this_df = this_df[this_df['mode'] == sl_mode].reset_index()
            if sl_key != 15:
                this_df = this_df[this_df['key'] == sl_key].reset_index()
            if sl_tempo != [0, 220]:
                this_df = this_df[(this_df['tempo'] >= sl_tempo[0]) & (this_df['tempo'] <= sl_tempo[1])].reset_index()

            # show arousal valence representations
            if av_rep == 'spoti_api':
                aro_col = 'energy'
                val_col = 'valence'
            elif av_rep == 'norm':
                aro_col = 'norm_energy'
                val_col = 'norm_valence'
            elif av_rep == 'users':
                # TODO!!!
                aro_col = 'energy'
                val_col = 'valence'

            if spoti_filt == 'none':
                this_color = None
                show_scale = False
            else:
                this_color = this_df[spoti_filt]
                show_scale = True

            text = 'Each point in the plot is a song, select one to view more information.'
            if click_data:
                click_point_np = np.array([click_data['points'][0][i] for i in ['x', 'y']]).astype(np.float64)
                bool_mask = (this_df.loc[:, [val_col, aro_col]].eq(click_point_np).all(axis=1))

                if bool_mask.any():
                    idx = this_df[bool_mask == True].index[0]

                    return self.make_song_info(idx, this_df)

            else:
                return 'Each point in the plot is a song, select one to view more information.'


if __name__ == "__main__":
    # instanciate Plotter
    plotter = Plotter()

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport",
                    "content": "width=device-width"}],
        external_stylesheets=external_stylesheets,
        routes_pathname_prefix='/',
        requests_pathname_prefix='/vis-mtg-mer/',
        serve_locally=False)

    server = app.server

    app.layout = plotter.create_layout(app)

    plotter.run_callbacks(app)
    app.run_server(host='0.0.0.0', debug=False)

