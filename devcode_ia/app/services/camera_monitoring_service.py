"""
Servicio de Monitoreo Continuo de Cámaras
Configurado para aceptar RTSP UDP de OBS
"""
import asyncio
import httpx
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import os

from app.services import get_detection_service
from app.models.multi_model_verifier import get_multi_model_verifier
from app.services.temporal_verifier import get_temporal_verifier
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)


class CameraMonitor:
    """
    Monitorea una cámara RTSP en tiempo real (UDP compatible)
    """

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        stream_url: str,
        latitude: float,
        longitude: float,
        webhook_url: str,
        confidence_threshold: float = 0.75,
        check_interval_seconds: int = 5
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.stream_url = stream_url
        self.latitude = latitude
        self.longitude = longitude
        self.webhook_url = webhook_url
        self.confidence_threshold = confidence_threshold
        self.check_interval = check_interval_seconds

        self.is_running = False
        self.frame_count = 0

        # Servicios
        logger.info(f"Inicializando servicios para {camera_id}...")
        try:
            self.detection_service = get_detection_service()
            if not hasattr(self.detection_service, 'detector') or self.detection_service.detector is None:
                logger.info("Inicializando modelo YOLOv8...")
                self.detection_service.initialize_model()
        except Exception as e:
            logger.error(f"Error inicializando detection_service: {e}")
            raise

        try:
            self.verifier = get_multi_model_verifier()
        except Exception as e:
            logger.warning(f"Verifier no disponible: {e}")
            self.verifier = None

        try:
            self.temporal_verifier = get_temporal_verifier(
                stream_id=camera_id,
                window_size=10,
                min_detections=5
            )
        except Exception as e:
            logger.warning(f"Temporal verifier no disponible: {e}")
            self.temporal_verifier = None

        try:
            self.storage = get_storage_service()
        except Exception as e:
            logger.warning(f"Storage no disponible: {e}")
            self.storage = None

        # Cliente HTTP para webhooks
        self.http_client = httpx.AsyncClient(timeout=30.0)

    def _create_video_capture(self) -> cv2.VideoCapture:
        """
        Crea un VideoCapture configurado para RTSP UDP

        Esta configuración es crucial para conectarse a streams RTSP que solo
        soportan UDP (como OBS configurado para RTSP UDP)
        """
        # Configurar variables de entorno FFmpeg para forzar UDP
        # Esto soluciona el error "461 Unsupported transport"
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

        # Crear VideoCapture con backend FFmpeg explícito
        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

        if cap.isOpened():
            # Configuraciones adicionales para mejor rendimiento con UDP
            # Buffer pequeño para reducir latencia con UDP
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            # Timeout reducido para UDP (ms)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)

            # Timeout de lectura reducido
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            logger.info(f"✅ Stream configurado con transporte UDP: {self.camera_name}")
        else:
            logger.error(f"❌ No se pudo abrir stream con UDP: {self.stream_url}")

        return cap

    async def start_monitoring(self):
        """Inicia el monitoreo de la cámara (UDP compatible)"""
        self.is_running = True
        logger.info(f"🎥 Iniciando monitoreo: {self.camera_name} ({self.camera_id})")
        logger.info(f"📡 Modo de transporte: RTSP sobre UDP")

        cap = None
        try:
            # Abrir stream con configuración UDP
            cap = self._create_video_capture()

            if not cap.isOpened():
                logger.error(f"❌ No se pudo abrir stream: {self.stream_url}")
                logger.error("💡 Verifica que:")
                logger.error("   1. La cámara/OBS esté transmitiendo en RTSP UDP")
                logger.error("   2. La URL RTSP sea correcta")
                logger.error("   3. El firewall permita tráfico UDP en el puerto RTSP")
                return

            logger.info(f"✅ Stream conectado: {self.camera_name}")

            # Loop de monitoreo
            no_frame_count = 0
            MAX_NO_FRAME = 30

            while self.is_running:
                ret, frame = cap.read()

                if not ret or frame is None:
                    no_frame_count += 1
                    if no_frame_count >= MAX_NO_FRAME:
                        logger.error("❌ Se perdió la conexión. Reintentando...")
                        cap.release()
                        await asyncio.sleep(2)
                        cap = self._create_video_capture()
                        no_frame_count = 0
                    await asyncio.sleep(0.05)
                    continue

                no_frame_count = 0
                self.frame_count += 1

                # Procesar cada 30 frames (~1 segundo a 30fps)
                if self.frame_count % 30 == 0:
                    logger.info(f"📸 Frame #{self.frame_count} - {self.camera_name}")
                    await self._process_frame(frame)

                await asyncio.sleep(0.033)  # ~30 FPS

        except asyncio.CancelledError:
            logger.info(f"🛑 Monitoreo cancelado: {self.camera_name}")
        except Exception as e:
            logger.error(f"❌ Error en monitoreo: {e}")
            logger.exception("Detalles del error:")
        finally:
            if cap is not None:
                cap.release()
            await self.http_client.aclose()
            logger.info(f"🛑 Monitoreo detenido: {self.camera_name}")

    async def _process_frame(self, frame: np.ndarray):
        """Procesa un frame del video"""
        try:
            frame_id = f"frame_{self.frame_count}"

            # 1. Detección con YOLOv8
            _, image_bytes = cv2.imencode('.jpg', frame)
            image_bytes = image_bytes.tobytes()

            detection_result = self.detection_service.predict_from_bytes(
                image_bytes=image_bytes,
                save_results=False
            )

            if detection_result.num_detections == 0:
                return

            logger.info(f"🔥 {self.camera_name}: {detection_result.num_detections} detecciones")

            # 2. Verificación multi-modelo (opcional)
            verified_detections = detection_result.detections
            if self.verifier:
                try:
                    verified_detections = self.verifier.verify_detections(
                        image=frame,
                        detections=detection_result.detections,
                        target_classes=['fire', 'smoke']
                    )
                    if len(verified_detections) == 0:
                        return
                except:
                    pass

            # 3. Verificación temporal (opcional)
            should_alert = True
            if self.temporal_verifier:
                try:
                    temporal_result = self.temporal_verifier.add_frame(
                        frame_id=frame_id,
                        detections=verified_detections
                    )
                    should_alert = temporal_result.get('should_alert', True)
                except:
                    pass

            # 4. Enviar webhook
            if should_alert:
                await self._send_webhook(
                    detection_result=detection_result,
                    verified_detections=verified_detections,
                    frame=frame,
                    frame_id=frame_id
                )

        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")

    async def _send_webhook(
        self,
        detection_result,
        verified_detections: List,
        frame: np.ndarray,
        frame_id: str
    ):
        """Envía webhook al backend"""
        try:
            # Preparar payload
            webhook_payload = {
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "confidence": max(d.confidence for d in verified_detections),
                "detected_at": datetime.now().isoformat(),
                "detections": [
                    {
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": {
                            "x1": d.bbox.x1,
                            "y1": d.bbox.y1,
                            "x2": d.bbox.x2,
                            "y2": d.bbox.y2
                        }
                    }
                    for d in verified_detections
                ],
                "verification_passed": True,
                "temporal_verified": True
            }

            # Enviar webhook
            logger.info(f"📤 Enviando webhook a: {self.webhook_url}")

            response = await self.http_client.post(
                self.webhook_url,
                json=webhook_payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 201:
                logger.info(f"✅ Webhook enviado: {self.camera_name}")
            else:
                logger.error(f"❌ Error webhook: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Error enviando webhook: {e}")

    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.is_running = False


class CameraMonitoringService:
    """Gestiona múltiples cámaras"""

    def __init__(self):
        self.monitors: Dict[str, CameraMonitor] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    def add_camera(
        self,
        camera_id: str,
        camera_name: str,
        stream_url: str,
        latitude: float,
        longitude: float,
        webhook_url: str
    ):
        """Agrega una cámara"""
        monitor = CameraMonitor(
            camera_id=camera_id,
            camera_name=camera_name,
            stream_url=stream_url,
            latitude=latitude,
            longitude=longitude,
            webhook_url=webhook_url
        )

        self.monitors[camera_id] = monitor
        logger.info(f"📹 Cámara agregada: {camera_name}")

    async def start_all(self):
        """Inicia todas las cámaras"""
        logger.info(f"🚀 Iniciando {len(self.monitors)} cámaras...")

        for camera_id, monitor in self.monitors.items():
            task = asyncio.create_task(monitor.start_monitoring())
            self.tasks[camera_id] = task

        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    def stop_all(self):
        """Detiene todas las cámaras"""
        logger.info("🛑 Deteniendo monitoreo...")
        for monitor in self.monitors.values():
            monitor.stop_monitoring()

    def get_status(self) -> Dict:
        """Estado de las cámaras"""
        return {
            camera_id: {
                "camera_name": monitor.camera_name,
                "is_running": monitor.is_running,
                "frames_processed": monitor.frame_count
            }
            for camera_id, monitor in self.monitors.items()
        }


# Instancia global
_monitoring_service: Optional[CameraMonitoringService] = None


def get_monitoring_service() -> CameraMonitoringService:
    """Obtiene instancia global"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = CameraMonitoringService()
    return _monitoring_service
