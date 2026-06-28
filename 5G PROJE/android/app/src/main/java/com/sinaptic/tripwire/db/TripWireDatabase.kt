package com.sinaptic.tripwire.db

import android.content.Context
import androidx.room.*
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.sinaptic.tripwire.model.AnalysisResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Yerel veritabanı entity'si (kritik tespitler için) */
@Entity(tableName = "detections")
data class DetectionEntity(
    @PrimaryKey
    val eventId:      String,
    val timestamp:    Long    = System.currentTimeMillis(),
    val riskScore:    Float   = 0f,
    val riskLevel:    String  = "DUSUK",
    val speedKmh:     Float?  = null,
    val behaviorClass: String? = null,
    val isQodActive:  Boolean = false,
)

/** DAO */
@Dao
interface DetectionDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: DetectionEntity): Long

    @Query("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 100")
    suspend fun getRecent(): List<DetectionEntity>

    @Query("DELETE FROM detections WHERE timestamp < :before")
    suspend fun deleteOlderThan(before: Long)

    @Query("SELECT COUNT(*) FROM detections WHERE riskLevel = 'KRITIK'")
    suspend fun getCriticalCount(): Int
}

/** Room Veritabanı */
@Database(entities = [DetectionEntity::class], version = 2, exportSchema = false)
abstract class TripWireDatabase : RoomDatabase() {
    abstract fun detectionDao(): DetectionDao

    companion object {
        @Volatile private var INSTANCE: TripWireDatabase? = null

        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """CREATE TABLE IF NOT EXISTS detections_new (
                        eventId TEXT NOT NULL PRIMARY KEY,
                        timestamp INTEGER NOT NULL,
                        riskScore REAL NOT NULL,
                        riskLevel TEXT NOT NULL,
                        speedKmh REAL,
                        behaviorClass TEXT,
                        isQodActive INTEGER NOT NULL
                    )""".trimIndent()
                )
                db.execSQL(
                    """INSERT INTO detections_new
                        (eventId, timestamp, riskScore, riskLevel, speedKmh, behaviorClass, isQodActive)
                        SELECT 'legacy-' || id, timestamp, riskScore, riskLevel,
                               speedKmh, behaviorClass, isQodActive FROM detections""".trimIndent()
                )
                db.execSQL("DROP TABLE detections")
                db.execSQL("ALTER TABLE detections_new RENAME TO detections")
            }
        }

        fun getInstance(context: Context): TripWireDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    TripWireDatabase::class.java,
                    "tripwire.db"
                ).addMigrations(MIGRATION_1_2)
                    .build().also { INSTANCE = it }
            }
    }
}

/** Repository — AnalysisResult'u entity'ye dönüştürür */
class DetectionRepository(context: Context) {
    private val dao = TripWireDatabase.getInstance(context).detectionDao()
    private val retentionMillis = 24L * 60L * 60L * 1000L

    suspend fun saveDetection(result: AnalysisResult, eventId: String? = null) = withContext(Dispatchers.IO) {
        dao.insert(
            DetectionEntity(
                eventId       = eventId ?: "local-${result.frameId}-${result.timestamp}",
                timestamp     = result.timestamp,
                riskScore     = result.riskScore,
                riskLevel     = result.riskLevel.name,
                // Canlı yerel audit, açık bir saklama politikası olmadan plaka tutmaz.
                speedKmh      = result.speedKmh,
                behaviorClass = result.behaviorClass,
                isQodActive   = result.isQodActive,
            )
        )
        dao.deleteOlderThan(System.currentTimeMillis() - retentionMillis)
    }

    suspend fun getRecentDetections() = withContext(Dispatchers.IO) {
        dao.getRecent()
    }
}
