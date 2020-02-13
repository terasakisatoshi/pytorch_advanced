# Dockerfile with PyTorch for my own purpose

# https://github.com/Idein/docker-pytorch
FROM idein/pytorch:v1.4.0-2020.02.07-2

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

USER ${NB_USER}
ENV PATH=${HOME}/.local/bin:$PATH

RUN pip install \
    jupyter \
    jupytext \
    ipywidgets \
    jupyter-contrib-nbextensions \
    jupyter-nbextensions-configurator \
    autopep8 --user

RUN mkdir ${HOME}/.jupyter && \
    echo "\
c.ContentsManager.default_jupytext_formats = 'ipynb,py'\n\
c.NotebookApp.contents_manager_class = 'jupytext.TextFileContentsManager'\n\
c.NotebookApp.open_browser = False\n\
\
" > ${HOME}/.jupyter/jupyter_notebook_config.py

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

# Make sure the contents of our repo are in ${HOME}
WORKDIR ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN pip install -r requirements.txt

EXPOSE 8888
