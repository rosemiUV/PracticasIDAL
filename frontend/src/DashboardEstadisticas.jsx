import React, { useState, useMemo } from 'react';
import { PieChart, Pie, Cell, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTooltip, Legend } from 'recharts';

export const MAPA_COLORES = {
  "Sumar": "#FF2D55",       // Apple Pink
  "PSOE": "#FF3B30",        // Apple Red
  "Vox": "#34C759",         // Apple Green
  "PP": "#007AFF",          // Apple Blue
  "Mesa": "#A2845E",        // Classic Gold/Brown
  "ERC": "#FFCC00",         // Apple Yellow
  "Junts": "#AF52DE",       // Apple Purple
  "EH Bildu": "#5AC8FA",    // Apple Cyan
  "PNV": "#00C7BE",         // Apple Teal
  "SIN PARTIDO": "#8E8E93"  // Apple Gray
};

// Función para formatear el tiempo
const formatearTiempo = (segundos) => {
  if (segundos < 60) {
    return `${segundos.toFixed(1)} segundos`;
  } else if (segundos < 3600) {
    const minutos = Math.floor(segundos / 60);
    const segsRestantes = Math.round(segundos % 60);
    return `${minutos} min ${segsRestantes} seg`;
  } else {
    const horas = Math.floor(segundos / 3600);
    const minutos = Math.floor((segundos % 3600) / 60);
    return `${horas} h ${minutos} min`;
  }
};

// Tooltip personalizado para la tarta
const CustomPieTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-[#1C1C1E]/90 backdrop-blur-md p-3 rounded-lg border border-white/10 text-white shadow-xl">
        <p className="font-semibold text-sm mb-1">{data.name}</p>
        <p className="text-xs text-white/70">Tiempo: {formatearTiempo(data.value)}</p>
      </div>
    );
  }
  return null;
};

const formatName = (name) => {
  if (!name) return '';
  if (name === 'DESCONOCIDO') return 'Desconocido';
  const parts = name.split(' ');
  if (parts.length > 1) {
    return `${parts[0].charAt(0)}. ${parts[parts.length - 1]}`;
  }
  return name;
};

// Tooltip personalizado para las barras
const CustomBarTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-[#1C1C1E]/90 backdrop-blur-md p-3 rounded-lg border border-white/10 text-white shadow-xl">
        <p className="font-semibold text-sm mb-1">{label} <span className="text-xs font-normal text-white/50">({data.partido})</span></p>
        <p className="text-xs text-[#0A84FF]">{payload[0].name}: {payload[0].value}%</p>
      </div>
    );
  }
  return null;
};

export default function DashboardEstadisticas({ data }) {
  const [filtroPartido, setFiltroPartido] = useState('Todos');

  const barras = data?.barras || [];
  const tarta = data?.tarta || [];

  // Extraer partidos únicos para los filtros (seguro porque barras es al menos [])
  const partidosDisponibles = ['Todos', ...Array.from(new Set(barras.map(b => b.partido)))];

  // Filtrar y preparar datos para el BarChart (useMemo DEBE ir antes de los returns tempranos)
  const barrasFiltradas = useMemo(() => {
    let filtradas = barras;
    if (filtroPartido !== 'Todos') {
      filtradas = barras.filter(b => b.partido === filtroPartido);
    }
    const mapeadas = filtradas.map(b => ({
      nombre: b.nombre,
      partido: b.partido,
      porcentaje_global: Number((b.porcentaje_global || 0).toFixed(1)),
      porcentaje_relativo: Number((b.porcentaje_relativo || 0).toFixed(1)),
      color: MAPA_COLORES[b.partido] || '#808080'
    }));
    
    // Ordenar de mayor a menor intervención
    return mapeadas.sort((a, b) => {
      const valA = filtroPartido === 'Todos' ? a.porcentaje_global : a.porcentaje_relativo;
      const valB = filtroPartido === 'Todos' ? b.porcentaje_global : b.porcentaje_relativo;
      return valB - valA;
    });
  }, [barras, filtroPartido]);

  if (!data || data.error) {
    return (
      <div className="flex justify-center items-center h-64 text-red-500 bg-red-500/10 rounded-xl">
        <p>{data?.error || "Error al cargar las estadísticas."}</p>
      </div>
    );
  }

  if (barras.length === 0 || tarta.length === 0) {
    return (
      <div className="flex justify-center items-center h-64 text-white/50 bg-white/5 rounded-xl border border-white/10 m-6">
        <div className="text-center">
          <p className="text-lg font-semibold text-white/70 mb-2">No hay datos disponibles</p>
          <p className="text-sm">El vídeo actual no tiene identificados ponentes o partidos.</p>
        </div>
      </div>
    );
  }

  // Preparar datos para el PieChart (puede ir aquí o antes, no es un hook)
  const datosTarta = [...tarta]
    .sort((a, b) => b.duracion - a.duracion) // Ordenar de mayor a menor tiempo
    .map(item => ({
      name: item.partido,
      value: item.duracion,
      color: MAPA_COLORES[item.partido] || '#808080'
    }));


  return (
    <div className="w-full h-full flex flex-col p-4 md:p-6 space-y-6 overflow-y-auto custom-scrollbar">
      <div className="flex items-center justify-between px-1">
        <h2 className="text-2xl font-semibold text-white/90 tracking-tight">Análisis de Tiempos de Habla</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-[400px]">
        {/* TARTA: Resumen por partidos */}
        <div className="bg-[#1C1C1E]/50 backdrop-blur-xl rounded-3xl p-6 border border-white/5 flex flex-col items-center shadow-2xl relative overflow-hidden">
          {/* Subtle top glare effect */}
          <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
          
          <h3 className="text-sm font-semibold text-white/60 mb-4 w-full text-left uppercase tracking-wider">Distribución por Partido</h3>
          
          <div className="flex-1 flex flex-col items-center justify-center w-full">
            <div className="w-full h-48 flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={datosTarta}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={85}
                    paddingAngle={6}
                    cornerRadius={12}
                    dataKey="value"
                    stroke="none"
                  >
                    {datosTarta.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} opacity={0.9} />
                    ))}
                  </Pie>
                  <RechartsTooltip content={<CustomPieTooltip />} cursor={{ fill: 'transparent' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            {/* Leyenda manual bajo la tarta */}
            <div className="w-full flex flex-wrap justify-center gap-x-4 gap-y-2 mt-4">
              {datosTarta.map((entry, index) => (
                 <div key={index} className="flex items-center text-xs font-medium text-white/70">
                   <span className="w-2.5 h-2.5 rounded-full mr-1.5 shadow-sm" style={{backgroundColor: entry.color}}></span>
                   {entry.name}
                 </div>
              ))}
            </div>
          </div>
        </div>

        {/* BARRAS: Detalle por ponente */}
        <div className="lg:col-span-2 bg-[#1C1C1E]/50 backdrop-blur-xl rounded-3xl p-6 border border-white/5 flex flex-col shadow-2xl relative overflow-hidden">
          {/* Subtle top glare effect */}
          <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
          
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
            <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider">
              {filtroPartido === 'Todos' ? 'Intervención Global' : `Intervención en ${filtroPartido}`}
            </h3>
            
            <div className="flex gap-2 mt-3 sm:mt-0 flex-wrap">
              {partidosDisponibles.map(partido => (
                <button
                  key={partido}
                  onClick={() => setFiltroPartido(partido)}
                  className={`px-3 py-1 text-xs rounded-full transition-all duration-300 ${
                    filtroPartido === partido
                      ? 'bg-white/10 text-white font-semibold backdrop-blur-md shadow-sm border border-white/10'
                      : 'bg-transparent text-white/40 hover:bg-white/5 hover:text-white/80 border border-transparent'
                  }`}
                >
                  {partido}
                </button>
              ))}
            </div>
          </div>

          <div className="w-full h-64 flex-1">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={barrasFiltradas}
                  margin={{ top: 10, right: 10, left: -25, bottom: 35 }}
                >
                  <CartesianGrid stroke="#ffffff08" vertical={false} horizontal={true} />
                  <XAxis 
                    dataKey="nombre" 
                    stroke="transparent" 
                    tick={{fill: '#ffffff60', fontSize: 10, fontWeight: 500}}
                    tickFormatter={formatName}
                    axisLine={false}
                    tickLine={false}
                    interval={0}
                    angle={-35}
                    textAnchor="end"
                    dy={5}
                    dx={-2}
                  />
                <YAxis 
                  stroke="transparent" 
                  tick={{fill: '#ffffff30', fontSize: 11}} 
                  axisLine={false}
                  tickLine={false}
                  domain={[0, 100]}
                />
                <BarTooltip cursor={{fill: 'rgba(255,255,255,0.03)'}} content={<CustomBarTooltip />} />
                <Bar 
                  dataKey={filtroPartido === 'Todos' ? "porcentaje_global" : "porcentaje_relativo"} 
                  name={filtroPartido === 'Todos' ? "% Global" : "% Relativo"} 
                  radius={[6, 6, 0, 0]}
                  barSize={filtroPartido !== 'Todos' && barrasFiltradas.length < 5 ? 40 : undefined}
                >
                  {barrasFiltradas.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} opacity={0.9} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
