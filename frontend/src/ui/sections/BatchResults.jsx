export default function BatchResults({ data }) {
  if (!Array.isArray(data) || data.length === 0) {
    return <div className="p-6 text-gray-500 italic">Sem dados para exibir.</div>;
  }

  // Mostra as principais colunas primeiro: p_planet, p_planet_float (se vier), e algumas features
  const cols = Object.keys(data[0]);

  // Colunas preferidas primeiro
  const preferredOrder = [
    "p_planet",
    "p_planet_float",
    "orig_idx",
    "koi_period",
    "koi_depth",
    "koi_incl",
    "st_teff",
    "st_logg",
  ];
  const orderedCols = [
    ...preferredOrder.filter((c) => cols.includes(c)),
    ...cols.filter((c) => !preferredOrder.includes(c)),
  ];

  return (
    <div className="overflow-auto">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-100 sticky top-0 z-10">
          <tr>
            {orderedCols.map((c) => (
              <th key={c} className="px-4 py-3 text-left font-semibold border-b">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr
              key={i}
              className={`border-b hover:bg-violet-50 ${
                // destaque suave para os mais prováveis
                typeof row.p_planet === "string" || typeof row.p_planet_float === "number"
                  ? ""
                  : ""
              }`}
            >
              {orderedCols.map((c) => (
                <td key={c} className="px-4 py-2 whitespace-nowrap">
                  {row[c] ?? "-"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="p-3 text-xs text-gray-500 border-t">
        Exibindo {data.length} linhas. (Ordenadas pelo backend por probabilidade decrescente)
      </div>
    </div>
  );
}