mkdir -p ~/.streamlit/
echo "
[general]n
email = "francisval.guedes.soares.037@ufrn.edu.br"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml