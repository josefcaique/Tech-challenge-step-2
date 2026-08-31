FROM python:3.11

ENV POETRY_VIRTUALENVS_CREATE=false
RUN pip3 install --no-cache-dir poetry==2.4.1

# ============================================
# 4. CAMADA DE CACHE (DEPENDÊNCIAS)
# ============================================
# ⚠️ IMPORTANTE: Este Dockerfile DEVE estar na RAIZ do projeto.
# Se estiver na raiz, o contexto de build enxerga pyproject.toml e poetry.lock.
COPY pyproject.toml /.
COPY poetry.lock /.


# Usa o poetry.lock já commitado (não roda `poetry lock` de novo aqui —
# reexecutar a resolução de dependências no build torna o build lento e
# não-reprodutível, já que pode resolver versões diferentes do lock commitado).
RUN poetry install --without dev --no-interaction --no-ansi --no-root

CMD ["/bin/bash"]
