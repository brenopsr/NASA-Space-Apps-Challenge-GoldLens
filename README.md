# NASA-Space-Apps-Challenge-GoldLens
Projeto criado para o desafio "A World Away: Hunting for Exoplanets with AI" do Nasa Space Apps Challenge 2025

## Nota sobre o backend
- **Problema:** o backend Flask não carregava automaticamente o modelo/arquivo de features distribuídos com o projeto, o que fazia o endpoint `/predict` falhar ao receber CSVs do frontend.
- **Solução:** foram adicionados caminhos padrão para localizar esses artefatos dentro do repositório e um fallback de decodificação para o relatório de métricas, garantindo que a API funcione imediatamente após o clone.
