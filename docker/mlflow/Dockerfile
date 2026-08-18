FROM python:3.11

RUN pip3 install --no-cache-dir poetry==2.4.1 && \
    poetry config virtualenvs.create false

# ============================================
# 4. CAMADA DE CACHE (DEPENDÊNCIAS)
# ============================================
# ⚠️ IMPORTANTE: Este Dockerfile DEVE estar na RAIZ do projeto.
# Se estiver na raiz, o contexto de build enxerga pyproject.toml e poetry.lock.
COPY pyproject.toml /. 
COPY poetry.lock /.


#RUN poetry config virtualenvs.create false \
#    && poetry install --no-interaction --no-ansi --without dev --no-root

#RUN poetry install

RUN poetry config virtualenvs.create false \
    && poetry lock \
    && poetry install --without dev --no-interaction --no-ansi --no-root


CMD ["/bin/bash"]