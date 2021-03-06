from plotly.graph_objs import YAxis,XAxis

THEMES = {
		'ggplot' : {
			'colorscale':'ggplot',
			'linewidth':1.3,
			'bargap' : .01,
			'layout' : {
				'legend' : {'bgcolor':'white','font':{'color':'grey10'}},
				'paper_bgcolor' : 'white',
				'plot_bgcolor' : 'grey14',
				'yaxis1' : YAxis({
					'tickfont' : {'color':'grey10'},
					'gridcolor' : 'lightivory',
					'titlefont' : {'color':'grey10'},
					'zerolinecolor' : 'lightivory',
					'showgrid' : True
				}),
				'xaxis1' : XAxis({
					'tickfont' : {'color':'grey10'},
					'gridcolor' : 'lightivory',
					'titlefont' : {'color':'grey10'},
					'zerolinecolor' : 'lightivory',
					'showgrid' : True
				}),
				'titlefont' : {'color':'charcoal'}
			},
			'annotations' : {
				'fontcolor' : 'grey10',
				'arrowcolor' : 'grey10'
			}

		},
		'pearl' : {
			'colorscale':'dflt',
			'linewidth':1.3,
			'bargap' : .01,
			'layout' : {
				'legend' : {'bgcolor':'pearl02','font':{'color':'pearl06'}},
				'paper_bgcolor' : 'pearl02',
				'plot_bgcolor' : 'pearl02',
				'yaxis1' : YAxis({
					'tickfont' : {'color':'pearl06'},
					'gridcolor' : 'pearl03',
					'titlefont' : {'color':'pearl06'},
					'zerolinecolor' : 'pearl03',
					'showgrid' : True
				}),
				'xaxis1' : XAxis({
					'tickfont' : {'color':'pearl06'},
					'gridcolor' : 'pearl03',
					'titlefont' : {'color':'pearl06'},
					'zerolinecolor' : 'pearl03',
					'showgrid' : True
				}),
				'titlefont' : {'color':'pearl06'}
			},
			'annotations' : {
				'fontcolor' : 'pearl06',
				'arrowcolor' : 'pearl04'
			},
			'3d' : {
				'yaxis1' : {
					'gridcolor' : 'pearl04',
					'zerolinecolor'  : 'pearl04'
				},
				'xaxis1' : {
					'gridcolor' : 'pearl04',
					'zerolinecolor'  : 'pearl04'
				}
			}
		},
		'solar' : {
			'colorscale':'dflt',
			'linewidth':1.3,
			'bargap' : .01,
			'layout' : {
				'legend' : {'bgcolor':'charcoal','font':{'color':'pearl'}},
				'paper_bgcolor' : 'charcoal',
				'plot_bgcolor' : 'charcoal',
				'yaxis1' : YAxis({
					'tickfont' : {'color':'grey12'},
					'gridcolor' : 'grey08',
					'titlefont' : {'color':'pearl'},
					'zerolinecolor' : 'grey09',
					'showgrid' : True
				}),
				'xaxis1' : XAxis({
					'tickfont' : {'color':'grey12'},
					'gridcolor' : 'grey08',
					'titlefont' : {'color':'pearl'},
					'zerolinecolor' : 'grey09',
					'showgrid' : True
				}),
				'titlefont' : {'color':'pearl'}
			},
			'annotations' : {
				'fontcolor' : 'pearl',
				'arrowcolor' : 'grey11'
			}
		},
		'space' : {
			'colorscale':'dflt',
			'linewidth':1.3,
			'bargap' : .01,
			'layout' : {
				'legend' : {'bgcolor':'grey03','font':{'color':'pearl'}},
				'paper_bgcolor' : 'grey03',
				'plot_bgcolor' : 'grey03',
				'yaxis1' : YAxis({
					'tickfont' : {'color':'grey12'},
					'gridcolor' : 'grey08',
					'titlefont' : {'color':'pearl'},
					'zerolinecolor' : 'grey09',
					'showgrid' : True
				}),
				'xaxis1' : XAxis({
					'tickfont' : {'color':'grey12'},
					'gridcolor' : 'grey08',
					'titlefont' : {'color':'pearl'},
					'zerolinecolor' : 'grey09',
					'showgrid' : True
				}),
				'titlefont' : {'color':'pearl'}
			},
			'annotations' : {
				'fontcolor' : 'pearl',
				'arrowcolor' : 'red'
			}
		},
		'white' : {
			'colorscale':'dflt',
			'linewidth':1.3,
			'bargap' : .01,
			'layout' : {
				'legend' : {'bgcolor':'white','font':{'color':'pearl06'}},
				'paper_bgcolor' : 'white',
				'plot_bgcolor' : 'white',
				'yaxis1' : YAxis({
					'tickfont' : {'color':'pearl06'},
					'gridcolor' : 'pearl03',
					'titlefont' : {'color':'pearl06'},
					'zerolinecolor' : 'pearl03',
					'showgrid' : True
				}),
				'xaxis1' : XAxis({
					'tickfont' : {'color':'pearl06'},
					'gridcolor' : 'pearl03',
					'titlefont' : {'color':'pearl06'},
					'zerolinecolor' : 'pearl03',
					'showgrid' : True
				}),
				'titlefont' : {'color':'pearl06'}
			},
			'annotations' : {
				'fontcolor' : 'pearl06',
				'arrowcolor' : 'pearl04'
			},
			'3d' : {
				'yaxis1' : {
					'gridcolor' : 'pearl04',
					'zerolinecolor'  : 'pearl04'
				},
				'xaxis1' : {
					'gridcolor' : 'pearl04',
					'zerolinecolor'  : 'pearl04'
				}
			}
		}
	}



