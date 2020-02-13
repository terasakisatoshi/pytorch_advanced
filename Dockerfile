# Dockerfile with PyTorch for my own purpose

# https://github.com/Idein/docker-pytorch
FROM idein/pytorch:v1.4.0-2020.02.07-2

RUN pip install \
    jupyter \
    jupytext \
    ipywidgets \
    jupyter-contrib-nbextensions \
    jupyter-nbextensions-configurator \
    autopep8
RUN mkdir /root/.jupyter && \
    echo "\
c.ContentsManager.default_jupytext_formats = 'ipynb,py'\n\
c.NotebookApp.open_browser = False\n\
\
" > /root/.jupyter/jupyter_notebook_config.py
# install/enable extension
RUN jupyter contrib nbextension install --user
RUN jupyter nbextensions_configurator enable --user
# enable extensions what you want
RUN jupyter nbextension enable code_prettify/autopep8
RUN jupyter nbextension enable select_keymap/main
RUN jupyter nbextension enable highlight_selected_word/main
RUN jupyter nbextension enable toggle_all_line_numbers/main
RUN jupyter nbextension enable varInspector/main
RUN jupyter nbextension enable execute_time/ExecuteTime

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888
