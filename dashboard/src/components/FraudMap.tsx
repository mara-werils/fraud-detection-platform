"use client";

import { useEffect } from "react";
import dynamic from "next/dynamic";
import type { ScoredTransaction } from "@/lib/types";

const MapContainerDynamic = dynamic(
  () => import("react-leaflet").then((mod) => mod.MapContainer),
  { ssr: false }
);
const TileLayerDynamic = dynamic(
  () => import("react-leaflet").then((mod) => mod.TileLayer),
  { ssr: false }
);
const CircleMarkerDynamic = dynamic(
  () => import("react-leaflet").then((mod) => mod.CircleMarker),
  { ssr: false }
);
const PopupDynamic = dynamic(
  () => import("react-leaflet").then((mod) => mod.Popup),
  { ssr: false }
);

interface FraudMapProps {
  transactions: ScoredTransaction[];
}

function getMarkerColor(score: number): string {
  if (score >= 0.7) return "#ef4444";
  if (score >= 0.3) return "#f97316";
  return "#22c55e";
}

function FraudMapInner({ transactions }: FraudMapProps) {
  useEffect(() => {
    // Import Leaflet CSS on client side
    // @ts-ignore
    import("leaflet/dist/leaflet.css");
  }, []);

  const markers = transactions
    .filter((tx) => tx.latitude && tx.longitude)
    .slice(0, 200);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-sm overflow-hidden">
      <div className="border-b border-gray-800 px-5 py-3.5">
        <h3 className="text-sm font-semibold text-white">
          Transaction Geography
        </h3>
      </div>
      <div className="h-[400px] w-full">
        <MapContainerDynamic
          center={[39.8283, -98.5795]}
          zoom={4}
          style={{ height: "100%", width: "100%", background: "#0a0f1a" }}
          zoomControl={false}
          attributionControl={false}
        >
          <TileLayerDynamic
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>'
          />
          {markers.map((tx) => (
            <CircleMarkerDynamic
              key={tx.txn_id}
              center={[tx.latitude, tx.longitude]}
              radius={5 + tx.fraud_score * 8}
              fillColor={getMarkerColor(tx.fraud_score)}
              color={getMarkerColor(tx.fraud_score)}
              weight={1}
              opacity={0.8}
              fillOpacity={0.5}
            >
              <PopupDynamic>
                <div className="text-xs space-y-1 text-gray-900 min-w-[160px]">
                  <div className="font-semibold">
                    {tx.txn_id.slice(0, 16)}...
                  </div>
                  <div>Amount: ${tx.amount.toFixed(2)}</div>
                  <div>Score: {(tx.fraud_score * 100).toFixed(0)}%</div>
                  <div>Decision: {tx.decision}</div>
                  <div>Merchant: {tx.merchant}</div>
                </div>
              </PopupDynamic>
            </CircleMarkerDynamic>
          ))}
        </MapContainerDynamic>
      </div>
      <div className="flex items-center gap-6 border-t border-gray-800 px-5 py-2.5">
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-green-500" />
          <span className="text-xs text-gray-400">Safe (&lt;0.3)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-orange-500" />
          <span className="text-xs text-gray-400">Warning (0.3-0.7)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
          <span className="text-xs text-gray-400">Fraud (&gt;0.7)</span>
        </div>
      </div>
    </div>
  );
}

const FraudMap = dynamic(() => Promise.resolve(FraudMapInner), { ssr: false });
export default FraudMap;
