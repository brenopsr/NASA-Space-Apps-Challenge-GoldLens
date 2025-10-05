import { useState, useMemo } from "react";
import BatchResults from "./BatchResults";

const API_URL = "http://localhost:5000/predict"; // ajuste se necessário

export default function BatchAnalysis() {
  const [loading, setLoading] = useState(false);
  const [csvFile, setCsvFile] = useState(null);
  const [results, setResults] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  // opções de formatação (padrões compatíveis com o backend)
  const [probFormat, setProbFormat] = useState("percent"); // "percent" | "float"
  const [probDecimals, setProbDecimals] = useState(2);
  const [includeIndex, setIncludeIndex] = useState(true);
  const [topN, setTopN] = useState(""); // vazio = sem limite

  const fileHint = useMemo(() => {
    if (!csvFile) return "";
    const mb = (csvFile.size / (1024 * 1024)).toFixed(2);
    return `${csvFile.name} • ${mb} MB`;
  }, [csvFile]);

  function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    const lower = file.name.toLowerCase();
    const allowed = lower.endsWith(".csv") || lower.endsWith(".xlsx") || lower.endsWith(".xls");
    if (!allowed) {
      setErrorMsg("Formato inválido. Use CSV, XLSX ou XLS.");
      setCsvFile(null);
      setResults(null);
      return;
    }
    setCsvFile(file);
    setResults(null);
    setErrorMsg("");
  }

  async function handleClassify() {
    if (!csvFile) return;
    setLoading(true);
    setErrorMsg("");
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("file", csvFile);

      const params = new URLSearchParams({
        format: "json",
        prob_format: probFormat,
        prob_decimals: String(probDecimals),
        include_index: includeIndex ? "1" : "0",
      });
      if (topN && Number(topN) > 0) params.set("top", String(Number(topN)));

      const response = await fetch(`${API_URL}?${params.toString()}`, {
        method: "POST",
        body: formData,
      });

      // tenta ler payload do backend (pode ser erro em JSON)
      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const msg = payload?.error || `Erro ${response.status}`;
        throw new Error(msg);
      }

      if (!Array.isArray(payload)) {
        throw new Error("Resposta inesperada do backend (esperado um array JSON).");
      }

      setResults(payload);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Erro ao classificar os dados.");
    } finally {
      setLoading(false);
    }
  }

  function handleExportCSV() {
    if (!results || results.length === 0) return;
    const headers = Object.keys(results[0]);
    const lines = [
      headers.join(","),
      ...results.map((row) =>
        headers
          .map((h) => {
            const v = row[h] ?? "";
            // escapando vírgula/aspas
            const s = String(v).replace(/"/g, '""');
            return /[",\n]/.test(s) ? `"${s}"` : s;
          })
          .join(",")
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `predicoes_${csvFile?.name?.replace(/\.(csv|xlsx|xls)$/i, "") || "saida"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="w-full max-w-5xl mx-auto">
      {/* Upload */}
      <div className="flex justify-center">
        <label
          htmlFor="csv-upload"
          className="w-full max-w-xl border-2 border-dashed border-gray-400 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer text-sm text-gray-600 mt-1 hover:text-white hover:border-white hover:bg-violet-500 transition"
        >
          <div className="text-5xl mb-3">📁</div>
          <p className="text-gray-800 font-medium">
            Arraste um arquivo <b>CSV/XLSX</b> (catálogo KOI/K2/TOI) ou clique para selecionar
          </p>
          <p className="mt-1">
            Ex.: <code>koi_period, koi_depth, koi_incl, st_teff, ...</code>
          </p>
          <input
            type="file"
            id="csv-upload"
            accept=".csv,.xlsx,.xls"
            hidden
            onChange={handleFileUpload}
          />
        </label>
      </div>

      {/* Painel de opções */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 border rounded-lg bg-white">
          <label className="block text-sm font-semibold mb-1">Formato da probabilidade</label>
          <select
            className="w-full border rounded px-3 py-2"
            value={probFormat}
            onChange={(e) => setProbFormat(e.target.value)}
          >
            <option value="percent">percent (ex.: 87,35%)</option>
            <option value="float">float (ex.: 0.8735)</option>
          </select>
        </div>

        <div className="p-4 border rounded-lg bg-white">
          <label className="block text-sm font-semibold mb-1">Casas decimais</label>
          <input
            type="number"
            min={0}
            max={8}
            className="w-full border rounded px-3 py-2"
            value={probDecimals}
            onChange={(e) => setProbDecimals(Math.min(8, Math.max(0, Number(e.target.value || 0))))}
          />
        </div>

        <div className="p-4 border rounded-lg bg-white">
          <label className="block text-sm font-semibold mb-1">Top N (opcional)</label>
          <input
            type="number"
            min={1}
            className="w-full border rounded px-3 py-2"
            placeholder="Ex.: 100"
            value={topN}
            onChange={(e) => setTopN(e.target.value)}
          />
          <label className="flex items-center gap-2 mt-3">
            <input
              type="checkbox"
              checked={includeIndex}
              onChange={(e) => setIncludeIndex(e.target.checked)}
            />
            <span className="text-sm">Incluir índice original</span>
          </label>
        </div>
      </div>

      {/* Status / Ações */}
      <div className="mt-6 p-4 border rounded-lg bg-white">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div className="text-sm text-gray-700">
            {csvFile ? (
              <span className="text-green-700 font-semibold">✅ Arquivo: {fileHint}</span>
            ) : (
              <span className="text-gray-500">Nenhum arquivo selecionado</span>
            )}
          </div>

          <div className="flex gap-3">
            <button
              disabled={!csvFile || loading}
              onClick={handleClassify}
              className={`px-5 py-2 rounded-lg text-white ${
                loading ? "bg-gray-400" : "bg-violet-600 hover:bg-violet-700"
              }`}
            >
              {loading ? "Classificando..." : "Classificar Dados"}
            </button>

            <button
              disabled={!results || results.length === 0}
              onClick={handleExportCSV}
              className={`px-5 py-2 rounded-lg border ${
                results && results.length > 0
                  ? "border-violet-600 text-violet-700 hover:bg-violet-50"
                  : "border-gray-300 text-gray-400 cursor-not-allowed"
              }`}
            >
              Exportar CSV
            </button>
          </div>
        </div>

        {!!errorMsg && (
          <div className="mt-3 p-3 rounded bg-red-50 text-red-700 text-sm border border-red-200">
            {errorMsg}
          </div>
        )}
      </div>

      {/* Resultados */}
      <div
        id="batch-results"
        className="mt-8 w-full bg-white shadow-sm rounded-xl p-0 border border-gray-200 overflow-hidden"
      >
        {loading ? (
          <div className="p-6 text-gray-600">Classificando dados...</div>
        ) : results && results.length > 0 ? (
          <BatchResults data={results} />
        ) : (
          <div className="p-6 text-gray-500 italic">Nenhum resultado ainda...</div>
        )}
      </div>
    </div>
  );
}
