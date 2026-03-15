limpiar:
	@echo "Limpiandoooo"
	cls

start:
	@echo "Iniciandoooo"
	$(MAKE) limpiar
	uv run streamlit run src/ui/app.py

ResetChromaDB:
	@echo "Reiniciando la base de datos de Chroma"
	$(MAKE) limpiar
	if exist data\chroma rmdir /s /q data\chroma
	uv run python src/rag/embedder.py
