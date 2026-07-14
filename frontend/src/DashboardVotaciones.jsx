import React, { useMemo } from 'react';
import { CheckCircle2, XCircle, MinusCircle, Vote, AlertCircle } from 'lucide-react';

// Tooltip para truncamiento si es necesario
const Tooltip = ({ children, text }) => (
  <div className="group relative inline-block">
    {children}
    <div className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 -translate-x-1/2 opacity-0 transition-opacity group-hover:opacity-100">
      <div className="whitespace-pre-wrap rounded-md bg-[#2C2C2E] px-3 py-2 text-sm text-white shadow-xl max-w-xs text-center">
        {text}
      </div>
      <div className="absolute left-1/2 top-full h-2 w-2 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-[#2C2C2E]"></div>
    </div>
  </div>
);

export default function DashboardVotaciones({ data }) {
  if (!data || data.error) {
    return (
      <div className="flex justify-center items-center h-64 text-red-500 bg-red-500/10 rounded-xl">
        <p>{data?.error || "Error al cargar las votaciones."}</p>
      </div>
    );
  }

  const votaciones = data.votaciones || [];
  const mensaje = data.mensaje;

  if (votaciones.length === 0) {
    return (
      <div className="flex justify-center items-center h-64 text-white/50 bg-[#1C1C1E]/50 rounded-xl border border-white/10 m-6 shadow-2xl">
        <div className="text-center flex flex-col items-center">
          <div className="bg-white/5 p-4 rounded-full mb-4">
            <Vote size={48} className="text-white/20" />
          </div>
          <p className="text-xl font-bold text-white/70 mb-2">Sin votaciones</p>
          <p className="text-sm">{mensaje || "No se detectaron votaciones durante esta sesión."}</p>
        </div>
      </div>
    );
  }

  const obtenerColorResolucion = (resolucion) => {
    const resLower = (resolucion || "").toLowerCase();
    if (resLower.includes("no se aprueba") || resLower.includes("no se admite") || resLower.includes("no se toma en consideración") || resLower.includes("no se convalida") || resLower.includes("se deroga")) return "text-red-400 bg-red-400/10 border-red-400/20";
    if (resLower.includes("aprueba") || resLower.includes("queda aprobad") || resLower.includes("se admite") || resLower.includes("se toma en consideración") || resLower.includes("se convalida") || resLower.includes("queda abocad")) return "text-green-400 bg-green-400/10 border-green-400/20";
    if (resLower.includes("empate")) return "text-orange-400 bg-orange-400/10 border-orange-400/20";
    return "text-blue-400 bg-blue-400/10 border-blue-400/20";
  };

  const obtenerIconoResolucion = (resolucion) => {
    const resLower = (resolucion || "").toLowerCase();
    if (resLower.includes("no se aprueba") || resLower.includes("no se admite") || resLower.includes("no se toma en consideración") || resLower.includes("no se convalida") || resLower.includes("se deroga")) return <XCircle size={16} />;
    if (resLower.includes("aprueba") || resLower.includes("queda aprobad") || resLower.includes("se admite") || resLower.includes("se toma en consideración") || resLower.includes("se convalida") || resLower.includes("queda abocad")) return <CheckCircle2 size={16} />;
    if (resLower.includes("empate")) return <AlertCircle size={16} />;
    return <MinusCircle size={16} />;
  };

  return (
    <div className="w-full h-full flex flex-col p-4 md:p-6 space-y-6 overflow-y-auto custom-scrollbar">
      <div className="flex items-center justify-between px-1">
        <h2 className="text-2xl font-bold text-white/90 tracking-tight flex items-center gap-2">
          <Vote className="text-blue-500" />
          Registro de Votaciones
        </h2>
        <span className="text-sm font-medium bg-white/10 text-white/70 px-3 py-1 rounded-full">
          {votaciones.length} votaciones detectadas
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {votaciones.map((vot, idx) => {
          const total = vot["Total"] || 0;
          const favor = vot["A favor"] || 0;
          const contra = vot["En contra"] || 0;
          const abs = vot["Abstenciones"] || 0;
          
          const maxVotes = Math.max(total, favor + contra + abs, 1);
          const pctFavor = (favor / maxVotes) * 100;
          const pctContra = (contra / maxVotes) * 100;
          const pctAbs = (abs / maxVotes) * 100;

          return (
            <div key={idx} className="bg-[#1C1C1E]/60 backdrop-blur-xl rounded-3xl p-5 border border-white/10 shadow-2xl flex flex-col relative overflow-hidden transition-all hover:border-white/20 hover:bg-[#1C1C1E]/80">
              {/* Subtle top glare effect */}
              <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
              
              <div className="flex justify-between items-start mb-4 gap-4">
                <h3 className="font-semibold text-white/90 text-sm leading-snug line-clamp-3" title={vot["Tema votado"]}>
                  {vot["Tema votado"] || "Tema Desconocido"}
                </h3>
                <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold border whitespace-nowrap shrink-0 ${obtenerColorResolucion(vot["Aprobado"])}`}>
                  {obtenerIconoResolucion(vot["Aprobado"])}
                  <span className="uppercase tracking-wider">{vot["Aprobado"] || "Resultado desconocido"}</span>
                </div>
              </div>

              <div className="mt-auto space-y-4">
                {/* Barras apiladas */}
                <div className="h-2 w-full flex rounded-full overflow-hidden bg-white/5 border border-white/5">
                  <div style={{ width: `${pctFavor}%` }} className="bg-green-500 h-full transition-all duration-500"></div>
                  <div style={{ width: `${pctContra}%` }} className="bg-red-500 h-full transition-all duration-500 border-l border-black/20"></div>
                  <div style={{ width: `${pctAbs}%` }} className="bg-yellow-500 h-full transition-all duration-500 border-l border-black/20"></div>
                </div>

                {/* Leyenda numérica */}
                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-white/5 rounded-xl p-2 text-center border border-green-500/10">
                    <p className="text-[10px] text-white/50 uppercase tracking-wider font-semibold mb-1">A favor</p>
                    <p className="text-lg font-bold text-green-400">{favor}</p>
                  </div>
                  <div className="bg-white/5 rounded-xl p-2 text-center border border-red-500/10">
                    <p className="text-[10px] text-white/50 uppercase tracking-wider font-semibold mb-1">En contra</p>
                    <p className="text-lg font-bold text-red-400">{contra}</p>
                  </div>
                  <div className="bg-white/5 rounded-xl p-2 text-center border border-yellow-500/10">
                    <p className="text-[10px] text-white/50 uppercase tracking-wider font-semibold mb-1">Abstención</p>
                    <p className="text-lg font-bold text-yellow-400">{abs}</p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
